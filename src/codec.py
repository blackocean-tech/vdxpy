"""4-color QR encoding library - Frame encoding/decoding with Reed-Solomon ECC"""
import math
import struct
import zlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
    _HAS_CREEDSOLO = True
except ImportError:
    from reedsolo import RSCodec, ReedSolomonError
    _HAS_CREEDSOLO = False

# Optimization D: RS syndrome check implemented in Cython (nogil compatible)
# Uses pre-built rs_fast.pyd if available. Falls back to Python numpy implementation otherwise.
try:
    import rs_fast
    _HAS_RS_FAST = True
except ImportError:
    _HAS_RS_FAST = False

from base import BaseCodec, FrameData
from config import ProfileConfig

# galois batch RS is wire-compatible with reedsolo (prim=0x11d, fcr=0, gen=2),
# achieving 22x speedup for decode_frame_from_bytes alone.
# However, startup cost is ~7s (import 0.8s + GF 1.3s + RS 3.5s + JIT first-run 1.4s),
# and since header decode (single block, batch disadvantage) dominates the current impl,
# galois is counterproductive for typical workloads (few MB) (measured 23s->32s).
# Only enabled when USE_GALOIS=1 env var is set (for large file processing).
import os as _os
_GALOIS_AVAILABLE = False
_galois = None
_GF256 = None
if _os.environ.get("USE_GALOIS"):
    try:
        import galois as _galois  # type: ignore
        _GF256 = _galois.GF(2**8, irreducible_poly=0x11d, primitive_element=2)
        _GALOIS_AVAILABLE = True
    except Exception:
        pass

ANCHOR_SIZE = 8  # cells
ANCHOR_PATTERN = np.array([
    [1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1],
    [1,0,1,0,0,1,0,1],
    [1,0,1,0,0,1,0,1],
    [1,0,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1],
], dtype=np.uint8)

HEADER_SIZE = 10  # v1 header: frame_num(4) + total_frames(2) + payload_len(2) + flags(1) + reserved(1)
HEADER_SIZE_V2 = 12  # v2 header: frame_num(4) + total_frames(2) + payload_len(4) + flags(1) + version(1)
# v2 required when: small cell_size causes per-frame payload > 65535 bytes
# v2 uses u32 for payload_len (max 4GB/frame)
RS_MAX_BLOCK = 255  # GF(2^8) max codeword size

# Optimization C: GF(2^8) tables for syndrome-only fast path
# creedsolo compatible: prim_poly=0x11d, generator=2, fcr=0
_GF_EXP: Optional[np.ndarray] = None  # shape (512,) uint8
_GF_LOG: Optional[np.ndarray] = None  # shape (256,) uint8
_GF_MUL_TABLE: Optional[np.ndarray] = None  # shape (256, 256) uint8


def _build_gf_tables(prim_poly: int = 0x11d):
    """Build GF(2^8) exp/log/mul tables on first call (singleton)."""
    global _GF_EXP, _GF_LOG, _GF_MUL_TABLE
    if _GF_MUL_TABLE is not None:
        return
    exp = np.zeros(512, dtype=np.uint8)
    log = np.zeros(256, dtype=np.uint8)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim_poly
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    mul_table = np.zeros((256, 256), dtype=np.uint8)
    for a in range(1, 256):
        la = int(log[a])
        for b in range(1, 256):
            mul_table[a, b] = exp[(la + int(log[b])) % 255]
    _GF_EXP, _GF_LOG, _GF_MUL_TABLE = exp, log, mul_table


# B1: zstd payload compression wrapper
# Decompresses if file_data (after extract_metadata) starts with ZSTD_MAGIC.
# Layout: ZSTD_MAGIC(4) | orig_size u64 BE | zstd compressed bytes
ZSTD_MAGIC = b"QRZ\x01"
ZSTD_HEADER_SIZE = 4 + 8


def wrap_zstd(data: bytes, level: int = 19) -> bytes:
    """Compress entire file_data with zstd and prepend magic header.
    level=19 is EM (below ultra) at ~tens of MB/s encode. Sender pre-compresses so no issue."""
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    return ZSTD_MAGIC + struct.pack(">Q", len(data)) + compressed


def unwrap_zstd(data: bytes) -> bytes:
    """Decompress if starts with ZSTD_MAGIC, otherwise return as-is."""
    if not data.startswith(ZSTD_MAGIC):
        return data
    if len(data) < ZSTD_HEADER_SIZE:
        raise ValueError("zstd-wrapped payload too short for header")
    orig_size = struct.unpack(">Q", data[4:ZSTD_HEADER_SIZE])[0]
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(data[ZSTD_HEADER_SIZE:], max_output_size=orig_size)
    if len(decompressed) != orig_size:
        raise ValueError(
            f"zstd decompress size mismatch: expected {orig_size}, got {len(decompressed)}"
        )
    return decompressed


class ColorQRCodec(BaseCodec):

    def __init__(self, config: ProfileConfig):
        super().__init__(config)
        self.cell_size = config.codec.cell_size
        self.num_colors = config.codec.num_colors
        self.bits_per_cell = int(math.log2(self.num_colors))
        self.colors = [tuple(c) for c in config.codec.colors]
        self.ecc_redundancy = config.codec.ecc_redundancy
        self.use_anchors = config.features.use_anchors
        self.use_ref_patches = config.features.reference_patches
        self.width, self.height = config.display.resolution

        self.grid_cols = self.width // self.cell_size
        self.grid_rows = self.height // self.cell_size

        self._build_cell_map()

        data_cells = len(self.data_cell_positions)
        raw_bits = data_cells * self.bits_per_cell
        self.frame_raw_bytes = raw_bits // 8

        # Pre-compute numpy arrays of cell center coordinates and reference colors for vectorization
        if self.data_cell_positions:
            pos_arr = np.array(self.data_cell_positions, dtype=np.int32)
            self._cell_y = pos_arr[:, 0] * self.cell_size + self.cell_size // 2
            self._cell_x = pos_arr[:, 1] * self.cell_size + self.cell_size // 2
            # Grid coordinates of data_cell_positions for sampling via cv2.resize
            self._dc_rows = pos_arr[:, 0]
            self._dc_cols = pos_arr[:, 1]
            # Whether data cells cover the entire grid (True if no anchors/ref_patches)
            self._dc_covers_all = (len(self.data_cell_positions) ==
                                    self.grid_rows * self.grid_cols)
        else:
            self._cell_y = np.array([], dtype=np.int32)
            self._cell_x = np.array([], dtype=np.int32)
            self._dc_rows = np.array([], dtype=np.int32)
            self._dc_cols = np.array([], dtype=np.int32)
            self._dc_covers_all = False
        # Store reference colors as BGR-order numpy array for vectorized distance calculation
        ref_bgr = np.array([(c[2], c[1], c[0]) for c in self.colors], dtype=np.int32)
        self._ref_bgr = ref_bgr  # shape (num_colors, 3)

        # Optimization: if all colors are BGR cube vertices (each component 0 or 255),
        # use BGR threshold lookup to avoid distance calculation (N*K*3 broadcast). ~2x speedup.
        self._bgr_lut = None
        if self._is_cube_vertex_palette():
            self._bgr_lut = self._build_bgr_lookup()

        # B2: if all colors are grayscale (R=G=B), further optimize with 256->cell_value LUT.
        # For use cases expressing 16 levels using Y-only without YUY2 chroma subsampling.
        self._gray_lut = None       # shape (256,) uint8 -- for BGR full-range
        self._gray_lut_yuy2 = None  # shape (256,) uint8 -- for YUY2 limited-range Y (#2)
        if self._is_grayscale_palette():
            self._gray_lut = self._build_gray_lookup()
            self._gray_lut_yuy2 = self._build_gray_lookup_yuy2()

        # Optimization #1: Y16 + chroma 1 bit (32 colors, 5 bits/cell)
        # First 16 colors are grayscale, last 16 are blue-biased with same Y.
        self._yc32_y_lut_yuy2 = None  # 256 -> 4-bit Y level
        self._yc32_cb_threshold = None
        if self._is_yc32_palette():
            self._yc32_y_lut_yuy2, self._yc32_cb_threshold = self._build_yc32_lookup()

        # Fingerprint sample positions for early known-frame detection
        # Extract N=32 positions at equal intervals from data cell area. Quantize each cell
        # to 1 byte via BGR threshold to produce a 32-byte fingerprint.
        if self.data_cell_positions:
            n_fp = min(32, len(self.data_cell_positions))
            step = max(1, len(self.data_cell_positions) // n_fp)
            fp_positions = self.data_cell_positions[::step][:n_fp]
            fp_arr = np.array(fp_positions, dtype=np.int32)
            self._fp_y = fp_arr[:, 0] * self.cell_size + self.cell_size // 2
            self._fp_x = fp_arr[:, 1] * self.cell_size + self.cell_size // 2
        else:
            self._fp_y = np.array([], dtype=np.int32)
            self._fp_x = np.array([], dtype=np.int32)

        # Encoder-side cache (finalized on first _encode_frame call)
        self._enc_palette = None
        self._enc_rs = None
        self._enc_cs = None
        # Per-frame RS encode result cache (significant speedup during loop playback)
        self._encoded_cache: dict = {}

        self.nsym = max(2, min(128, int(RS_MAX_BLOCK * self.ecc_redundancy)))
        self.rs = RSCodec(self.nsym)
        self.rs_data_per_block = RS_MAX_BLOCK - self.nsym

        if self.frame_raw_bytes >= RS_MAX_BLOCK:
            self.num_blocks = self.frame_raw_bytes // RS_MAX_BLOCK
            self.rs_block_size = RS_MAX_BLOCK
        else:
            self.num_blocks = 1
            self.nsym = max(2, int(self.frame_raw_bytes * self.ecc_redundancy))
            self.rs = RSCodec(self.nsym)
            self.rs_data_per_block = self.frame_raw_bytes - self.nsym
            self.rs_block_size = self.frame_raw_bytes

        # header v2 detection: auto-enable if explicitly enabled or if small cell_size exceeds u16 limit
        max_v1_payload = 65535
        provisional = self.num_blocks * self.rs_data_per_block - HEADER_SIZE
        auto_v2 = provisional > max_v1_payload
        self._use_v2 = bool(self.config.codec.header_v2) or auto_v2
        self._header_size = HEADER_SIZE_V2 if self._use_v2 else HEADER_SIZE
        self._header_fmt = ">IHIBB" if self._use_v2 else ">IHHBB"
        self.payload_capacity = max(
            1, self.num_blocks * self.rs_data_per_block - self._header_size
        )

        # Optimization #5: partial sample scanning only the first RS block (rs_block_size bytes)
        # Calculate cell positions after rs_block_size and bits_per_cell are determined
        if self.bits_per_cell > 0 and self.data_cell_positions:
            cells_for_first_block = (self.rs_block_size * 8) // self.bits_per_cell
            cells_for_first_block = min(cells_for_first_block,
                                        len(self.data_cell_positions))
            hdr_cells = self.data_cell_positions[:cells_for_first_block]
            hdr_arr = np.array(hdr_cells, dtype=np.int32)
            self._hdr_cell_y = hdr_arr[:, 0] * self.cell_size + self.cell_size // 2
            self._hdr_cell_x = hdr_arr[:, 1] * self.cell_size + self.cell_size // 2
            self._hdr_cell_count = cells_for_first_block
        else:
            self._hdr_cell_y = np.array([], dtype=np.int32)
            self._hdr_cell_x = np.array([], dtype=np.int32)
            self._hdr_cell_count = 0

        # Optimization C: Allocate GF(2^8) tables on first codec construction
        _build_gf_tables()
        # alpha_powers[j] = alpha^j (j=0..nsym-1) for syndrome calc
        self._gf_alpha_powers = _GF_EXP[:self.nsym].copy() if _GF_EXP is not None else None
        # Disable fast path with SYNDROME_FAST=0 env var (for regression testing)
        self._use_syndrome_fast = (
            _os.environ.get("SYNDROME_FAST", "1") != "0"
            and _GF_MUL_TABLE is not None
        )

        # galois fast RS (auto-use if installed)
        # Wire-compatible with reedsolo (prim=0x11d, fcr=0, gen=2), no sender changes needed
        self._galois_rs = None
        if _GALOIS_AVAILABLE and self.rs_data_per_block < self.rs_block_size:
            try:
                self._galois_rs = _galois.ReedSolomon(
                    self.rs_block_size, self.rs_data_per_block,
                    field=_GF256, c=0
                )
            except Exception:
                self._galois_rs = None

    def _build_cell_map(self):
        reserved = set()

        if self.use_anchors:
            for corner_r, corner_c in self._anchor_corners():
                for dr in range(ANCHOR_SIZE):
                    for dc in range(ANCHOR_SIZE):
                        reserved.add((corner_r + dr, corner_c + dc))

        if self.use_ref_patches:
            corners = [
                (0, 0), (0, self.grid_cols - 2),
                (self.grid_rows - 2, 0), (self.grid_rows - 2, self.grid_cols - 2)
            ]
            for r0, c0 in corners:
                for dr in range(2):
                    for dc in range(2):
                        reserved.add((r0 + dr, c0 + dc))

        self.data_cell_positions = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if (r, c) not in reserved:
                    self.data_cell_positions.append((r, c))

    def _anchor_corners(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),
            (0, self.grid_cols - ANCHOR_SIZE),
            (self.grid_rows - ANCHOR_SIZE, 0),
        ]

    def _color_to_value(self, bgr: Tuple[int, int, int]) -> int:
        min_dist = float('inf')
        best = 0
        for i, ref in enumerate(self.colors):
            ref_bgr = (ref[2], ref[1], ref[0])
            dist = sum((a - b) ** 2 for a, b in zip(bgr, ref_bgr))
            if dist < min_dist:
                min_dist = dist
                best = i
        return best

    def _value_to_color_bgr(self, val: int) -> Tuple[int, int, int]:
        rgb = self.colors[val % len(self.colors)]
        return (rgb[2], rgb[1], rgb[0])

    def _rs_encode(self, data: bytes, num_blocks: int) -> bytes:
        padded = data.ljust(num_blocks * self.rs_data_per_block, b'\x00')
        # Use galois batch encode if available (21x speedup)
        if self._galois_rs is not None:
            try:
                arr = np.frombuffer(padded, dtype=np.uint8).reshape(
                    num_blocks, self.rs_data_per_block
                )
                encoded = self._galois_rs.encode(arr)
                return bytes(encoded.tobytes())
            except Exception:
                pass  # fallback
        # Fallback: reedsolo per-block
        blocks = []
        for i in range(num_blocks):
            start = i * self.rs_data_per_block
            block = padded[start:start + self.rs_data_per_block]
            encoded = bytes(self.rs.encode(block))
            blocks.append(encoded)
        return b"".join(blocks)

    def _compute_syndromes_batch(self, codewords: np.ndarray) -> np.ndarray:
        """Optimization C: Batch compute syndromes for codewords (N, n).
        Preprocessing to determine error presence without calling creedsolo's rs.decode.
        Parallelizes Horner's method across both block axis N and syndrome axis nsym (255 numpy ops).
        Returns: (N, nsym) uint8 array.
        """
        N, n = codewords.shape
        nsym = self.nsym
        mul_table = _GF_MUL_TABLE
        alpha_powers = self._gf_alpha_powers  # (nsym,) uint8
        # Initial value: s[k, j] = codewords[k, 0] (broadcast across nsym)
        s = np.broadcast_to(codewords[:, 0:1], (N, nsym)).copy()
        # Horner's method: s = mul(s, alpha_j) + codewords[:, i] (XOR in GF)
        # Each step is advanced indexing + XOR on (N, nsym) shaped arrays
        col_i = codewords[:, 1:]  # shape (N, n-1)
        for i in range(n - 1):
            s = mul_table[s, alpha_powers[None, :]] ^ col_i[:, i:i+1]
        return s

    def _rs_decode_fast(self, data: bytes, num_blocks: int) -> Optional[bytes]:
        """Optimization C/D: Full RS batch decode via Cython rs_fast.
        Phase 2: clean blocks use syndrome check only, error blocks use BM+Chien+Forney correction.
        All processing runs within nogil with zero Python loops or object creation.
        Falls back to Python numpy probe + per-block creedsolo when rs_fast is unavailable.
        """
        expected_len = num_blocks * self.rs_block_size
        if len(data) < expected_len:
            data = data.ljust(expected_len, b'\x00')
        arr = np.frombuffer(data[:expected_len], dtype=np.uint8).reshape(
            num_blocks, self.rs_block_size
        )

        if _HAS_RS_FAST:
            # Phase 2: Full Cython batch decode (syndrome + BM + Chien + Forney)
            arr_c = np.ascontiguousarray(arr)
            decoded, success = rs_fast.decode_blocks_batch(arr_c, self.nsym)
            if not np.all(success):
                return None
            return bytes(decoded.tobytes())

        # Fallback: Python numpy probe + per-block creedsolo
        N_PROBE = 8
        ABORT_THRESHOLD = 0.5
        probe_n = min(N_PROBE, num_blocks)
        probe_syn = self._compute_syndromes_batch(arr[:probe_n])
        probe_has_error = (probe_syn != 0).any(axis=1)
        if int(probe_has_error.sum()) / probe_n > ABORT_THRESHOLD:
            return None
        if probe_n < num_blocks:
            rest_syn = self._compute_syndromes_batch(arr[probe_n:])
            has_error_rest = (rest_syn != 0).any(axis=1)
            has_error = np.concatenate([probe_has_error, has_error_rest])
        else:
            has_error = probe_has_error

        out_blocks = []
        for i in range(num_blocks):
            if not has_error[i]:
                out_blocks.append(arr[i, :self.rs_data_per_block].tobytes())
            else:
                block = bytes(arr[i])
                try:
                    result = self.rs.decode(block)
                    decoded = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
                    out_blocks.append(decoded)
                except Exception:
                    return None
        return b"".join(out_blocks)

    def _rs_decode_chunk(self, rs, data: bytes, n_blocks: int) -> Optional[bytes]:
        """C: Per-worker function for parallel decode. Decodes num_blocks consecutively and returns combined result."""
        out = []
        for i in range(n_blocks):
            block = data[i * self.rs_block_size:(i + 1) * self.rs_block_size]
            if len(block) < self.rs_block_size:
                block = block.ljust(self.rs_block_size, b'\x00')
            try:
                result = rs.decode(block)
                decoded = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
                out.append(decoded)
            except Exception:
                return None
        return b"".join(out)

    def _rs_decode_parallel(self, data: bytes, num_blocks: int,
                            workers: int) -> Optional[bytes]:
        """C: Split RS blocks within a frame across workers for parallel decode.
        Theoretically Nx if creedsolo releases GIL.
        Uses dedicated RSCodec per worker (to avoid internal state races),
        submits via ThreadPoolExecutor and joins.
        """
        from concurrent.futures import ThreadPoolExecutor
        if not hasattr(self, "_rs_pool") or len(self._rs_pool) < workers:
            self._rs_pool = [RSCodec(self.nsym) for _ in range(workers)]
        if not hasattr(self, "_rs_executor"):
            self._rs_executor = ThreadPoolExecutor(max_workers=workers)

        # Each worker handles a contiguous group of blocks
        per_worker = (num_blocks + workers - 1) // workers
        futures = []
        idx = 0
        for w in range(workers):
            n = min(per_worker, num_blocks - idx)
            if n <= 0:
                break
            chunk_data = data[idx * self.rs_block_size:(idx + n) * self.rs_block_size]
            fut = self._rs_executor.submit(
                self._rs_decode_chunk, self._rs_pool[w], chunk_data, n
            )
            futures.append(fut)
            idx += n

        results = []
        for fut in futures:
            decoded = fut.result()
            if decoded is None:
                return None
            results.append(decoded)
        return b"".join(results)

    def _rs_decode(self, data: bytes, num_blocks: int) -> Optional[bytes]:
        # Optimization C: syndrome-only fast path (skip creedsolo for clean blocks)
        # Can be disabled with SYNDROME_FAST=0 env var
        if self._use_syndrome_fast and num_blocks > 5:
            result = self._rs_decode_fast(data, num_blocks)
            if result is not None:
                return result
            # If fast path returns None, fall through to normal path
        # Optimization C': RS_PARALLEL=N env var (default 1) splits RS blocks within a frame
        # across N worker threads. Since creedsolo doesn't release GIL during decode,
        # 1 thread is usually fastest, but kept for future GIL-release implementations
        rs_par = int(_os.environ.get("RS_PARALLEL", "1") or "1")
        if rs_par > 1 and num_blocks > rs_par * 2:
            return self._rs_decode_parallel(data, num_blocks, rs_par)
        # If galois is available, batch decode all blocks (18x speedup)
        if self._galois_rs is not None:
            try:
                expected_len = num_blocks * self.rs_block_size
                if len(data) < expected_len:
                    data = data.ljust(expected_len, b'\x00')
                arr = np.frombuffer(data[:expected_len], dtype=np.uint8).reshape(
                    num_blocks, self.rs_block_size
                )
                # Check for uncorrectable errors with decode(codeword, errors=True)
                decoded, n_errors = self._galois_rs.decode(arr, errors=True)
                # Rows with n_errors == -1 are uncorrectable (per block)
                if np.any(np.asarray(n_errors) < 0):
                    return None
                return bytes(np.asarray(decoded).tobytes())
            except Exception:
                # Fall back to per-block on batch failure
                pass
        # Fallback: reedsolo per-block
        blocks = []
        for i in range(num_blocks):
            start = i * self.rs_block_size
            block = data[start:start + self.rs_block_size]
            if len(block) < self.rs_block_size:
                block = block.ljust(self.rs_block_size, b'\x00')
            try:
                result = self.rs.decode(block)
                decoded = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
                blocks.append(decoded)
            except Exception:
                return None
        return b"".join(blocks)

    # Optimization #4: LT (fountain) code mode
    LT_PREFIX_SIZE = 12  # lt-code: filesize(4) + blocksize(4) + blockseed(4)

    def prepare_chunks_lt(self, file_path: str,
                          oversampling: float = 1.5,
                          compress: bool = False,
                          compress_level: int = 19,
                          ) -> List[Tuple[int, bytes, int, bool]]:
        """Generate frame sequence using LT (Luby Transform) fountain code.
        Each frame becomes an LT-encoded block to resolve coupon collector tail stalling.
        Receiver can reconstruct from any K + epsilon blocks regardless of playback order.
        """
        import os, io
        import lt.encode as lte

        with open(file_path, "rb") as f:
            file_data = f.read()
        if compress:
            file_data = wrap_zstd(file_data, level=compress_level)

        filename = os.path.basename(file_path)
        filename_bytes = filename.encode("utf-8")
        data = struct.pack(">H", len(filename_bytes)) + filename_bytes + file_data

        # 1 LT block = LT_PREFIX_SIZE + blocksize. Must fit within payload_capacity.
        blocksize = self.payload_capacity - self.LT_PREFIX_SIZE
        if blocksize <= 0:
            raise ValueError(
                f"payload_capacity {self.payload_capacity} too small for LT (need >= {self.LT_PREFIX_SIZE+1})"
            )
        K = (len(data) + blocksize - 1) // blocksize
        M = max(int(K * oversampling), K + 5)  # at least 5 blocks of margin

        encoder = lte.encoder(io.BytesIO(data), blocksize)
        chunks: List[Tuple[int, bytes, int, bool]] = []
        for i in range(M):
            blk = next(encoder)
            # blk size is LT_PREFIX_SIZE + blocksize = payload_capacity
            chunks.append((i, blk, M, i == M - 1))
        return chunks

    def prepare_chunks(self, file_path: str, compress: bool = False,
                       compress_level: int = 19) -> List[Tuple[int, bytes, int, bool]]:
        """Split file into payload chunks (frame images are not generated yet).

        Since frames are not held as images, memory usage is only the file body even for large files.
        Returns list of (frame_num, payload, total_frames, is_last).
        Adds parity frames if FEC is configured (fec_M_per_group > 0).
        compress=True pre-compresses file_data with zstd and sends with magic header.
        """
        import os
        with open(file_path, "rb") as f:
            file_data = f.read()

        if compress:
            file_data = wrap_zstd(file_data, level=compress_level)

        filename = os.path.basename(file_path)
        filename_bytes = filename.encode("utf-8")
        data = struct.pack(">H", len(filename_bytes)) + filename_bytes + file_data

        # FEC requires all data chunks to be equal length (payload_capacity),
        # so short trailing chunks are zero-padded (receiver recovers last_payload_len from frame header)
        chunks_data = []
        offset = 0
        while offset < len(data):
            chunks_data.append(data[offset:offset + self.payload_capacity])
            offset += self.payload_capacity
        if not chunks_data:
            chunks_data = [b""]

        K = len(chunks_data)
        M = self.config.codec.fec_M_per_group
        G = self.config.codec.fec_group_size

        if M <= 0:
            # No FEC
            total = K
            return [(i, c, total, i == total - 1) for i, c in enumerate(chunks_data)]

        # FEC enabled: encode chunks_data per group_size via encode_with_fec_chunked
        # Equalize all data chunks to equal length (prerequisite for FEC encode)
        equal_chunks = [
            c.ljust(self.payload_capacity, b"\x00") for c in chunks_data
        ]
        from frame_fec import encode_with_fec_chunked
        all_chunks = encode_with_fec_chunked(equal_chunks, M, group_size=G)
        # First K items of all_chunks are data (order preserved), then M parity interleaved
        # encode_with_fec_chunked output order:
        # [g0_data..., g0_parity..., g1_data..., g1_parity..., ...]
        n_groups = (K + G - 1) // G
        total = K + M * n_groups

        # is_last_data is True for the last DATA frame (K-th frame)
        # is_last for parity frames is False
        # Receiver recovers actual payload length from (frame_num, len(payload), is_last)
        result = []
        cursor = 0
        cur_data_idx = 0
        for g in range(n_groups):
            k_g = min(G, K - g * G)
            # data frames
            for i in range(k_g):
                payload = all_chunks[cursor]
                cursor += 1
                # Only the last data chunk retains actual payload length (others are equal to payload_capacity)
                is_last_data = (cur_data_idx == K - 1)
                if is_last_data:
                    payload = chunks_data[-1]  # restore original length
                result.append((cur_data_idx, payload, total, is_last_data))
                cur_data_idx += 1
            # parity frames
            for i in range(M):
                parity = all_chunks[cursor]
                cursor += 1
                # parity frame_num = K + g*M + i
                fn = K + g * M + i
                result.append((fn, parity, total, False))
        return result

    def encode_chunk(self, frame_num: int, payload: bytes,
                     total_frames: int, is_last: bool) -> np.ndarray:
        """Generate a single frame image (for on-demand invocation)."""
        return self._encode_frame(frame_num, payload, total_frames, is_last)

    def encode_chunk_to_cells(self, frame_num: int, payload: bytes,
                              total_frames: int, is_last: bool) -> np.ndarray:
        """Generate up to RS encode + cell value grid (grid_rows, grid_cols) uint8.
        Does not generate image (skips cv2.resize/palette). For M3 intermediate representation."""
        cache_key = (frame_num, total_frames, is_last)
        encoded = self._encoded_cache.get(cache_key)
        if encoded is None:
            flags = 0x01 if is_last else 0x00
            version = 2 if self._use_v2 else 0
            header = struct.pack(self._header_fmt, frame_num, total_frames,
                                 len(payload), flags, version)
            raw = header + payload
            encoded = self._rs_encode(raw, self.num_blocks)
            self._encoded_cache[cache_key] = encoded
        cell_values = self._bytes_to_cells_array(encoded)
        # Initialize palette/coordinate cache on first call
        if self._enc_palette is None:
            self._enc_palette = np.array(
                [(c[2], c[1], c[0]) for c in self.colors], dtype=np.uint8
            )
            if self.data_cell_positions:
                pos = np.array(self.data_cell_positions, dtype=np.int32)
                self._enc_rs = pos[:, 0]
                self._enc_cs = pos[:, 1]
            else:
                self._enc_rs = np.array([], dtype=np.int32)
                self._enc_cs = np.array([], dtype=np.int32)
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        n = min(len(cell_values), len(self._enc_rs))
        if n > 0:
            grid[self._enc_rs[:n], self._enc_cs[:n]] = cell_values[:n].astype(np.uint8)
        return grid

    def cell_grid_to_image(self, grid: np.ndarray) -> np.ndarray:
        """cell grid (uint8, grid_rows x grid_cols) -> display image (uint8, height x width x 3).
        Only palette lookup + cv2.resize NEAREST. Fast since no RS encode needed."""
        if self._enc_palette is None:
            self._enc_palette = np.array(
                [(c[2], c[1], c[0]) for c in self.colors], dtype=np.uint8
            )
        color_grid = self._enc_palette[grid]  # (gr, gc, 3) uint8
        target_h = self.grid_rows * self.cell_size
        target_w = self.grid_cols * self.cell_size
        scaled = cv2.resize(color_grid, (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST)
        if scaled.shape[0] == self.height and scaled.shape[1] == self.width:
            img = scaled
        else:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:scaled.shape[0], :scaled.shape[1]] = scaled
        if self.use_anchors:
            self._draw_anchors(img)
        if self.use_ref_patches:
            self._draw_reference_patches(img)
        return img

    def encode_file(self, file_path: str) -> List[np.ndarray]:
        """Return all frames as a list (for small files; may run out of memory for large files).

        For large files, use the streaming API: prepare_chunks + encode_chunk.
        """
        chunks = self.prepare_chunks(file_path)
        return [self._encode_frame(*c) for c in chunks]

    def _encode_frame(self, frame_num: int, payload: bytes,
                      total_frames: int, is_last: bool) -> np.ndarray:
        # Cache RS-encoded byte sequence (significant speedup from 70ms to 0ms during loop playback)
        cache_key = (frame_num, total_frames, is_last)
        encoded = self._encoded_cache.get(cache_key)
        if encoded is None:
            flags = 0x01 if is_last else 0x00
            version = 2 if self._use_v2 else 0
            header = struct.pack(self._header_fmt, frame_num, total_frames,
                                 len(payload), flags, version)
            raw = header + payload
            encoded = self._rs_encode(raw, self.num_blocks)
            self._encoded_cache[cache_key] = encoded

        cell_values = self._bytes_to_cells_array(encoded)  # numpy ndarray

        # Vectorized: build cell grid -> palette lookup -> scale up by cell_size
        if self._enc_palette is None:
            self._enc_palette = np.array(
                [(c[2], c[1], c[0]) for c in self.colors], dtype=np.uint8
            )
            if self.data_cell_positions:
                pos = np.array(self.data_cell_positions, dtype=np.int32)
                self._enc_rs = pos[:, 0]
                self._enc_cs = pos[:, 1]
            else:
                self._enc_rs = np.array([], dtype=np.int32)
                self._enc_cs = np.array([], dtype=np.int32)

        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        n = min(len(cell_values), len(self._enc_rs))
        if n > 0:
            grid[self._enc_rs[:n], self._enc_cs[:n]] = cell_values[:n]

        color_grid = self._enc_palette[grid]  # shape (grid_rows, grid_cols, 3)
        # Scale up by cell_size using cv2.resize NEAREST interpolation (~5x faster than np.repeat)
        target_h = self.grid_rows * self.cell_size
        target_w = self.grid_cols * self.cell_size
        import cv2 as _cv2
        scaled = _cv2.resize(color_grid, (target_w, target_h),
                             interpolation=_cv2.INTER_NEAREST)

        if scaled.shape[0] == self.height and scaled.shape[1] == self.width:
            img = scaled
        else:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:scaled.shape[0], :scaled.shape[1]] = scaled

        if self.use_anchors:
            self._draw_anchors(img)

        if self.use_ref_patches:
            self._draw_reference_patches(img)

        return img

    def _bytes_to_cells_array(self, data: bytes) -> np.ndarray:
        """Convert data byte sequence to numpy array of cell values (vectorized)."""
        if not data:
            return np.array([], dtype=np.int32)
        arr = np.frombuffer(data, dtype=np.uint8)
        bpc = self.bits_per_cell
        # B (post #5): bpc=4 dedicated fast path (1 byte = 2 cells, high 4bit + low 4bit)
        # ~5x faster than general path (unpackbits)
        if bpc == 4:
            cells = np.empty(arr.size * 2, dtype=np.int32)
            cells[0::2] = (arr >> 4) & 0xF
            cells[1::2] = arr & 0xF
            return cells
        bits = np.unpackbits(arr)  # MSB first
        n_bits = len(bits)
        pad = (-n_bits) % bpc
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        chunks = bits.reshape(-1, bpc).astype(np.int32)
        powers = (1 << np.arange(bpc - 1, -1, -1)).astype(np.int32)
        return (chunks * powers).sum(axis=1).astype(np.int32)

    def _cells_to_bytes_array(self, cells, num_bytes: int) -> bytes:
        """Convert cell value array to byte sequence (vectorized).
        Avoids O(N^2) big-int accumulation from Python version for bits_per_cell=3.
        bits_per_cell=3 has dedicated fast path (8 cells x 3 bits = 24 bits = 3 bytes).
        """
        if isinstance(cells, np.ndarray):
            cell_arr = cells.astype(np.uint8)
        else:
            cell_arr = np.asarray(cells, dtype=np.uint8)
        if cell_arr.size == 0:
            return b""
        bpc = self.bits_per_cell

        # Dedicated fast path: bits_per_cell=3 divides evenly as 8 cells = 24 bits = 3 bytes
        # Direct uint32 packing per 8-cell group, no unpackbits/packbits needed
        if bpc == 3:
            n = cell_arr.size
            pad = (-n) % 8
            if pad:
                cell_arr = np.concatenate([cell_arr, np.zeros(pad, dtype=np.uint8)])
            grouped = cell_arr.reshape(-1, 8).astype(np.uint32)
            combined = ((grouped[:, 0] << 21) | (grouped[:, 1] << 18)
                        | (grouped[:, 2] << 15) | (grouped[:, 3] << 12)
                        | (grouped[:, 4] << 9) | (grouped[:, 5] << 6)
                        | (grouped[:, 6] << 3) | grouped[:, 7])
            out = np.empty(combined.size * 3, dtype=np.uint8)
            out[0::3] = (combined >> 16) & 0xFF
            out[1::3] = (combined >> 8) & 0xFF
            out[2::3] = combined & 0xFF
            return bytes(out[:num_bytes])

        # B (post #5): bpc=4 dedicated fast path (2 cells = 8 bits = 1 byte)
        if bpc == 4:
            n = cell_arr.size
            if n % 2:
                cell_arr = np.concatenate([cell_arr, np.zeros(1, dtype=np.uint8)])
            pairs = cell_arr.reshape(-1, 2)
            out = ((pairs[:, 0] << 4) | (pairs[:, 1] & 0xF)).astype(np.uint8)
            return bytes(out[:num_bytes])

        # General path: create bit array and packbits
        bits = np.empty(cell_arr.size * bpc, dtype=np.uint8)
        for i in range(bpc):
            bits[i::bpc] = (cell_arr >> (bpc - 1 - i)) & 1
        n_bits = len(bits)
        pad = (-n_bits) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        packed = np.packbits(bits)
        return bytes(packed[:num_bytes])

    def _bytes_to_cells(self, data: bytes) -> List[int]:
        cells = []
        if self.bits_per_cell == 1:
            for byte in data:
                for bit in range(7, -1, -1):
                    cells.append((byte >> bit) & 1)
        elif self.bits_per_cell == 2:
            for byte in data:
                cells.append((byte >> 6) & 0x03)
                cells.append((byte >> 4) & 0x03)
                cells.append((byte >> 2) & 0x03)
                cells.append(byte & 0x03)
        elif self.bits_per_cell == 3:
            bit_buffer = 0
            bit_count = 0
            for byte in data:
                bit_buffer = (bit_buffer << 8) | byte
                bit_count += 8
                while bit_count >= 3:
                    bit_count -= 3
                    cells.append((bit_buffer >> bit_count) & 0x07)
        elif self.bits_per_cell == 4:
            for byte in data:
                cells.append((byte >> 4) & 0x0F)
                cells.append(byte & 0x0F)
        return cells

    def _cells_to_bytes(self, cells: List[int], num_bytes: int) -> bytes:
        if self.bits_per_cell == 1:
            result = bytearray()
            for i in range(0, len(cells), 8):
                byte = 0
                for j in range(8):
                    if i + j < len(cells):
                        byte = (byte << 1) | (cells[i + j] & 1)
                    else:
                        byte <<= 1
                result.append(byte)
            return bytes(result[:num_bytes])
        elif self.bits_per_cell == 2:
            result = bytearray()
            for i in range(0, len(cells), 4):
                byte = 0
                for j in range(4):
                    if i + j < len(cells):
                        byte = (byte << 2) | (cells[i + j] & 0x03)
                    else:
                        byte <<= 2
                result.append(byte)
            return bytes(result[:num_bytes])
        elif self.bits_per_cell == 3:
            bit_buffer = 0
            bit_count = 0
            result = bytearray()
            for val in cells:
                bit_buffer = (bit_buffer << 3) | (val & 0x07)
                bit_count += 3
                while bit_count >= 8:
                    bit_count -= 8
                    result.append((bit_buffer >> bit_count) & 0xFF)
            return bytes(result[:num_bytes])
        elif self.bits_per_cell == 4:
            result = bytearray()
            for i in range(0, len(cells), 2):
                hi = cells[i] & 0x0F
                lo = cells[i + 1] & 0x0F if i + 1 < len(cells) else 0
                result.append((hi << 4) | lo)
            return bytes(result[:num_bytes])
        return b""

    def _draw_anchors(self, img: np.ndarray):
        white = (255, 255, 255)
        black = (0, 0, 0)
        for corner_r, corner_c in self._anchor_corners():
            for dr in range(ANCHOR_SIZE):
                for dc in range(ANCHOR_SIZE):
                    color = white if ANCHOR_PATTERN[dr, dc] else black
                    y = (corner_r + dr) * self.cell_size
                    x = (corner_c + dc) * self.cell_size
                    img[y:y+self.cell_size, x:x+self.cell_size] = color

    def _draw_reference_patches(self, img: np.ndarray):
        corners = [
            (0, 0), (0, self.grid_cols - 2),
            (self.grid_rows - 2, 0), (self.grid_rows - 2, self.grid_cols - 2)
        ]
        patch_colors = [(0, 0, 0), (255, 255, 255), (0, 0, 0), (255, 255, 255)]
        for (r0, c0), color in zip(corners, patch_colors):
            for dr in range(2):
                for dc in range(2):
                    y = (r0 + dr) * self.cell_size
                    x = (c0 + dc) * self.cell_size
                    img[y:y+self.cell_size, x:x+self.cell_size] = color

    def _is_cube_vertex_palette(self) -> bool:
        """Check if all colors are BGR cube vertices (each component is 0 or 255)."""
        for rgb in self.colors:
            for v in rgb:
                if v != 0 and v != 255:
                    return False
        return True

    def _build_bgr_lookup(self) -> Optional[np.ndarray]:
        """Build BGR threshold bits -> cell value lookup table.
        Returns None if ambiguous (duplicate bit patterns).
        """
        lut = np.full(8, -1, dtype=np.int8)
        for idx, rgb in enumerate(self.colors):
            r, g, b = rgb
            bits = ((1 if b >= 128 else 0) << 2) | ((1 if g >= 128 else 0) << 1) | (1 if r >= 128 else 0)
            if lut[bits] != -1:
                return None  # duplicate found
            lut[bits] = idx
        return lut.astype(np.uint8)

    def _is_grayscale_palette(self) -> bool:
        """Check if all colors are grayscale with R=G=B (e.g., 16-level Y-only palette)."""
        if len(self.colors) < 2:
            return False
        for rgb in self.colors:
            if not (rgb[0] == rgb[1] == rgb[2]):
                return False
        return True

    def _build_gray_lookup(self) -> np.ndarray:
        """256-entry LUT from grayscale 0..255 pixel value -> nearest level index.
        Levels need not be equally spaced (selects minimum distance level for each 0..255 value).
        """
        # levels: luminance values of grayscale palette (use R since R==G==B)
        levels = np.array([rgb[0] for rgb in self.colors], dtype=np.int32)  # shape (K,)
        all_v = np.arange(256, dtype=np.int32)[:, None]  # shape (256, 1)
        dist = np.abs(all_v - levels[None, :])  # shape (256, K)
        nearest = np.argmin(dist, axis=1).astype(np.uint8)  # shape (256,)
        return nearest

    def _is_yc32_palette(self) -> bool:
        """Optimization #1: Check for 32-color structure with first 16 grayscale, last 16 blue-biased.
        Blue-bias adds 1 bit by shifting B > R (Cb shift) while preserving Y.
        """
        if len(self.colors) != 32:
            return False
        # First 16 colors are grayscale (R == G == B)
        for i in range(16):
            r, g, b = self.colors[i]
            if not (r == g == b):
                return False
        # Last 16 colors are blue-biased (B > R)
        for i in range(16, 32):
            r, g, b = self.colors[i]
            if not (b > r):
                return False
        # Index pairs (i, i+16) should have same Y (not checked at encode time, noted for reference)
        return True

    def _build_yc32_lookup(self) -> Tuple[np.ndarray, int]:
        """Map 16 levels (R values of palette[0..15] as L) to YUY2 limited Y.
        Also returns Cb threshold (boundary value for chroma=1 determination).
        """
        # R values of first 16 colors (should be R==G==B)
        levels_full = np.array([self.colors[i][0] for i in range(16)], dtype=np.int32)
        # YUY2 limited Y = 16 + L * 219/255
        levels_yuy2 = np.round(16 + levels_full * 219.0 / 255.0).astype(np.int32)
        all_v = np.arange(256, dtype=np.int32)[:, None]
        dist = np.abs(all_v - levels_yuy2[None, :])
        nearest = np.argmin(dist, axis=1).astype(np.uint8)

        # Cb threshold: gray has Cb=128, blue-bias has ~145 -> midpoint 136
        # In YUY2 Cb is also limited range, but neutral 128 is near center.
        cb_threshold = 136
        return nearest, cb_threshold

    def _build_gray_lookup_yuy2(self) -> np.ndarray:
        """Optimization #2: YUY2 limited-range Y value -> nearest level index LUT.
        Palette R values encoded in BGR full-range (0..255) are sent/received via
        HDMI YUY2 (BT.601 limited range) as Y' = 16 + R * 219/255.
        This LUT is for direct Y channel reference from raw YUY2 capture (shape (H,W,2)).
        """
        # Map palette R (= G = B for grayscale) to YUY2 limited-range Y
        levels_yuy2 = np.array(
            [round(16 + rgb[0] * 219.0 / 255.0) for rgb in self.colors],
            dtype=np.int32,
        )
        all_v = np.arange(256, dtype=np.int32)[:, None]
        dist = np.abs(all_v - levels_yuy2[None, :])
        nearest = np.argmin(dist, axis=1).astype(np.uint8)
        return nearest

    def _sample_cells(self, image: np.ndarray) -> Tuple[bytes, np.ndarray]:
        """Extract cell values from image and return raw_bytes and cell_values array."""
        h_img, w_img = image.shape[:2]
        cs = self.cell_size
        n_ch = image.shape[2] if image.ndim >= 3 else 1

        # Phase 4: Cython integrated path (cell extract + byte packing in C loop)
        in_range = (h_img >= self.grid_rows * cs and w_img >= self.grid_cols * cs)
        if _HAS_RS_FAST and in_range and self._dc_covers_all:
            if self._gray_lut_yuy2 is not None and n_ch == 2:
                raw_bytes = rs_fast.sample_cells_gray_lut_yuy2(
                    image, cs, self.grid_rows, self.grid_cols,
                    self._gray_lut_yuy2, self.bits_per_cell)
                return raw_bytes, np.empty(0, dtype=np.uint8)
            elif self._gray_lut is not None and n_ch == 3:
                raw_bytes = rs_fast.sample_cells_gray_lut(
                    image, cs, self.grid_rows, self.grid_cols,
                    self._gray_lut, 1, self.bits_per_cell)
                return raw_bytes, np.empty(0, dtype=np.uint8)
            elif self._bgr_lut is not None and n_ch == 3:
                raw_bytes = rs_fast.sample_cells_bgr_lut(
                    image, cs, self.grid_rows, self.grid_cols,
                    self._bgr_lut, self.bits_per_cell)
                return raw_bytes, np.empty(0, dtype=np.uint8)

        # Fallback: numpy path
        # Fastest path: extract cell center pixels as zero-copy view via stride slicing.
        # (fancy index 0.39ms -> stride view 0.0002ms = ~2000x faster view access)
        # Subsequent threshold + LUT also run in 2D.
        if (self._yc32_y_lut_yuy2 is not None and in_range and n_ch == 2):
            # Optimization #1: YC32 (Y16 + chroma 1 bit) path
            # If cs (cell_size) is even and cs//2 is also even, chroma sample position = Cb
            # cs=4: center x = 2 (even, Cb), cs=8: center x = 4 (even, Cb)
            small_y = image[cs // 2:self.grid_rows * cs:cs,
                            cs // 2:self.grid_cols * cs:cs, 0]
            small_chroma = image[cs // 2:self.grid_rows * cs:cs,
                                 cs // 2:self.grid_cols * cs:cs, 1]
            y_idx = self._yc32_y_lut_yuy2[small_y]               # 0..15
            chroma_bit = (small_chroma >= self._yc32_cb_threshold).astype(np.uint8)
            cv_grid = (chroma_bit << 4) | y_idx                  # 0..31
            if self._dc_covers_all:
                cell_values_arr = cv_grid.ravel()
            else:
                cell_values_arr = cv_grid[self._dc_rows, self._dc_cols]
            raw_byte_count = (len(self.data_cell_positions) * self.bits_per_cell) // 8
            raw_bytes = self._cells_to_bytes_array(cell_values_arr, raw_byte_count)
            return raw_bytes, cell_values_arr
        if (self._gray_lut_yuy2 is not None and in_range and n_ch == 2):
            # Optimization #2: directly receive raw YUY2 (H, W, 2) and reference Y channel only
            small_y = image[cs // 2:self.grid_rows * cs:cs,
                            cs // 2:self.grid_cols * cs:cs, 0]
            cv_grid = self._gray_lut_yuy2[small_y]
            if self._dc_covers_all:
                cell_values_arr = cv_grid.ravel()
            else:
                cell_values_arr = cv_grid[self._dc_rows, self._dc_cols]
            raw_byte_count = (len(self.data_cell_positions) * self.bits_per_cell) // 8
            raw_bytes = self._cells_to_bytes_array(cell_values_arr, raw_byte_count)
            return raw_bytes, cell_values_arr
        if self._gray_lut is not None and in_range and n_ch == 3:
            # B2: Y-only grayscale path (1 channel LUT)
            # mean(axis=2) averages 3 channels for noise robustness but slower than G-only.
            # For YUY2 input where R=G=B, G channel alone is sufficient.
            small_g = image[cs // 2:self.grid_rows * cs:cs,
                            cs // 2:self.grid_cols * cs:cs, 1]  # G channel only
            cv_grid = self._gray_lut[small_g]
            if self._dc_covers_all:
                cell_values_arr = cv_grid.ravel()
            else:
                cell_values_arr = cv_grid[self._dc_rows, self._dc_cols]
        elif (self._bgr_lut is not None and in_range):
            # image[cs//2::cs, cs//2::cs, :] is a (grid_rows, grid_cols, 3) view of cell center pixels
            small = image[cs // 2:self.grid_rows * cs:cs,
                          cs // 2:self.grid_cols * cs:cs, :]
            # threshold + LUT (cell values are in 0..255 range, kept as uint8)
            b_bit = (small[:, :, 0] >= 128).astype(np.uint8) << 2
            g_bit = (small[:, :, 1] >= 128).astype(np.uint8) << 1
            r_bit = (small[:, :, 2] >= 128).astype(np.uint8)
            bits_grid = b_bit | g_bit | r_bit
            cv_grid = self._bgr_lut[bits_grid]
            if self._dc_covers_all:
                cell_values_arr = cv_grid.ravel()  # kept as uint8 (Q5)
            else:
                cell_values_arr = cv_grid[self._dc_rows, self._dc_cols]
        else:
            # Fallback path: fancy index
            ys = np.clip(self._cell_y, 0, h_img - 1)
            xs = np.clip(self._cell_x, 0, w_img - 1)
            if self._gray_lut is not None:
                sampled = image[ys, xs]  # (N, 3)
                cell_values_arr = self._gray_lut[sampled[:, 1]].astype(np.int32)
            elif self._bgr_lut is not None:
                sampled = image[ys, xs]
                b_bit = (sampled[:, 0] >= 128).astype(np.uint8) << 2
                g_bit = (sampled[:, 1] >= 128).astype(np.uint8) << 1
                r_bit = (sampled[:, 2] >= 128).astype(np.uint8)
                bits = b_bit | g_bit | r_bit
                cell_values_arr = self._bgr_lut[bits].astype(np.int32)
            else:
                sampled = image[ys, xs].astype(np.int32)
                diff = sampled[:, None, :] - self._ref_bgr[None, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                cell_values_arr = np.argmin(dist_sq, axis=1).astype(np.int32)

        raw_byte_count = (len(self.data_cell_positions) * self.bits_per_cell) // 8
        raw_bytes = self._cells_to_bytes_array(cell_values_arr, raw_byte_count)
        return raw_bytes, cell_values_arr

    def compute_fingerprint(self, image: np.ndarray) -> bytes:
        """Lightweight fingerprint for early known-frame detection. Samples only 32 cells
        and quantizes to 32 bytes via BGR threshold. Much faster than _sample_cells (~7ms).
        """
        if self._fp_y.size == 0:
            return b""
        # _fp_y, _fp_x are center pixel coordinates of data_cell_positions
        sampled = image[self._fp_y, self._fp_x]
        n_ch = image.shape[2] if image.ndim >= 3 else 1
        if n_ch == 2:
            # Optimization #2: raw YUY2 input. Y channel (index 0) upper 4-bit + Cb 1 bit
            # for 5-bit fingerprint. 32 cells x 5 bits = 160-bit hash space is sufficient.
            y_bits = (sampled[:, 0] >> 4).astype(np.uint8)
            if self._yc32_y_lut_yuy2 is not None:
                # YC32: incorporate Cb into fingerprint
                cb_bit = (sampled[:, 1] >= 136).astype(np.uint8) << 4
                return (cb_bit | y_bits).tobytes()
            return y_bits.tobytes()
        # BGR path (existing)
        b_bit = (sampled[:, 0] >= 128).astype(np.uint8) << 2
        g_bit = (sampled[:, 1] >= 128).astype(np.uint8) << 1
        r_bit = (sampled[:, 2] >= 128).astype(np.uint8)
        return (b_bit | g_bit | r_bit).tobytes()

    def _sample_header_cells(self, image: np.ndarray) -> bytes:
        """Optimization #5: partial sample scanning only cells for the first RS block
        (rs_block_size bytes). Extracts only ~510 needed cells instead of full frame
        _sample_cells (~1ms), so 100x+ faster."""
        if self._hdr_cell_count == 0:
            return b""
        n_ch = image.shape[2] if image.ndim >= 3 else 1
        if self._yc32_y_lut_yuy2 is not None and n_ch == 2:
            # #1 YC32 (Y16 + chroma 1 bit) partial sample
            y_sample = image[self._hdr_cell_y, self._hdr_cell_x, 0]
            cb_sample = image[self._hdr_cell_y, self._hdr_cell_x, 1]
            y_idx = self._yc32_y_lut_yuy2[y_sample]
            chroma_bit = (cb_sample >= self._yc32_cb_threshold).astype(np.uint8)
            cells = (chroma_bit << 4) | y_idx
        elif self._gray_lut_yuy2 is not None and n_ch == 2:
            sampled = image[self._hdr_cell_y, self._hdr_cell_x, 0]
            cells = self._gray_lut_yuy2[sampled]
        elif self._gray_lut is not None and n_ch == 3:
            sampled = image[self._hdr_cell_y, self._hdr_cell_x, 1]
            cells = self._gray_lut[sampled]
        elif self._bgr_lut is not None and n_ch == 3:
            sampled = image[self._hdr_cell_y, self._hdr_cell_x]
            b_bit = (sampled[:, 0] >= 128).astype(np.uint8) << 2
            g_bit = (sampled[:, 1] >= 128).astype(np.uint8) << 1
            r_bit = (sampled[:, 2] >= 128).astype(np.uint8)
            cells = self._bgr_lut[b_bit | g_bit | r_bit]
        else:
            # General path (slow). Delegate to full path without header-specific path.
            return b""
        # cells (uint8) -> bytes (rs_block_size length)
        raw = self._cells_to_bytes_array(cells, self.rs_block_size)
        return raw

    def decode_frame_header(self, image: np.ndarray,
                            fast: bool = True) -> Optional[Tuple[int, int, bytes]]:
        """Decode only the first RS block to quickly obtain frame_num.
        Used to identify already-received frames. ~50x faster (47 blocks -> 1 block).
        fast=True (#5): scan only required cells to extract header.
                        For full decode after new frame detection, call decode_frame_full
                        or decode_frame_from_bytes(image=) separately.
        Returns: (frame_num, total_frames, raw_bytes) or None.
        raw_bytes is b"" when fast=True (use decode_frame_full if full sample needed).
        """
        if self.use_anchors and self.config.features.perspective_correction:
            image = self._correct_perspective(image)
        if self.use_ref_patches and self.config.features.color_normalization:
            image = self._normalize_colors(image)

        if fast and self._hdr_cell_count > 0:
            # #5: partial sample path
            block = self._sample_header_cells(image)
            if len(block) >= self.rs_block_size:
                try:
                    result = self.rs.decode(block[:self.rs_block_size])
                    decoded = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
                except Exception:
                    decoded = None
                if decoded and len(decoded) >= self._header_size:
                    try:
                        header_unpacked = struct.unpack(
                            self._header_fmt, decoded[:self._header_size]
                        )
                    except struct.error:
                        return None
                    frame_num, total_frames, payload_len = header_unpacked[:3]
                    max_payload = self.payload_capacity
                    if (0 < total_frames <= 0xFFFF
                            and frame_num < total_frames
                            and payload_len <= max_payload):
                        # Don't return raw_bytes (fast path). If caller needs it,
                        # delegate full sample to decode_frame_full(image) or
                        # decode_frame_from_bytes.
                        return (frame_num, total_frames, b"")
            # fast path failed -> fall through to full path

        raw_bytes, _ = self._sample_cells(image)

        # Decode only the first RS block to obtain header
        block = raw_bytes[:self.rs_block_size]
        if len(block) < self.rs_block_size:
            block = block.ljust(self.rs_block_size, b'\x00')
        try:
            result = self.rs.decode(block)
            decoded = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
        except Exception:
            return None

        if len(decoded) < self._header_size:
            return None
        frame_num, total_frames, payload_len, _flags, _ = struct.unpack(
            self._header_fmt, decoded[:self._header_size]
        )
        if total_frames == 0 or total_frames > 0xFFFF:
            return None
        if frame_num >= total_frames:
            return None
        max_payload = self.num_blocks * self.rs_data_per_block - self._header_size
        if payload_len > max_payload:
            return None
        return frame_num, total_frames, raw_bytes

    def decode_frame_from_bytes(self, raw_bytes: bytes) -> Optional[FrameData]:
        """Full decode from raw_bytes already extracted by _sample_cells (skips cell sampling)."""
        decoded = self._rs_decode(raw_bytes, self.num_blocks)
        if decoded is None or len(decoded) < self._header_size:
            return None
        frame_num, total_frames, payload_len, flags, _ = struct.unpack(
            self._header_fmt, decoded[:self._header_size]
        )
        if total_frames == 0 or total_frames > 0xFFFF:
            return None
        if frame_num >= total_frames:
            return None
        max_payload = self.num_blocks * self.rs_data_per_block - self._header_size
        if payload_len > max_payload:
            return None
        payload = decoded[self._header_size:self._header_size + payload_len]
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        is_last = bool(flags & 0x01)
        return FrameData(
            frame_number=frame_num,
            payload=payload,
            crc=crc,
            total_frames=total_frames,
            is_last=is_last,
        )

    def decode_frame(self, image: np.ndarray) -> Optional[FrameData]:
        """Obtain FrameData from image. Leverages fast paths via _sample_cells
        including BGR threshold LUT and stride view.
        """
        if self.use_anchors and self.config.features.perspective_correction:
            image = self._correct_perspective(image)
        if self.use_ref_patches and self.config.features.color_normalization:
            image = self._normalize_colors(image)
        raw_bytes, _ = self._sample_cells(image)
        return self.decode_frame_from_bytes(raw_bytes)

    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        import cv2
        h, w = image.shape[:2]
        anchor_positions = self._anchor_corners()
        if len(anchor_positions) < 3:
            return image

        src_points = []
        dst_points = []
        for corner_r, corner_c in anchor_positions:
            expected_cx = (corner_c + ANCHOR_SIZE / 2) * self.cell_size
            expected_cy = (corner_r + ANCHOR_SIZE / 2) * self.cell_size
            detected = self._find_anchor_center(image, int(expected_cy), int(expected_cx))
            if detected is None:
                return image
            src_points.append(detected)
            dst_points.append((expected_cx, expected_cy))

        fourth_src_x = src_points[2][0] + (src_points[1][0] - src_points[0][0])
        fourth_src_y = src_points[1][1] + (src_points[2][1] - src_points[0][1])
        src_points.append((fourth_src_x, fourth_src_y))

        fourth_dst_x = dst_points[2][0] + (dst_points[1][0] - dst_points[0][0])
        fourth_dst_y = dst_points[1][1] + (dst_points[2][1] - dst_points[0][1])
        dst_points.append((fourth_dst_x, fourth_dst_y))

        src = np.float32(src_points)
        dst = np.float32(dst_points)

        max_deviation = np.max(np.abs(src - dst))
        if max_deviation < self.cell_size * ANCHOR_SIZE:
            return image

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h))

    def _find_anchor_center(self, image: np.ndarray, approx_y: int,
                            approx_x: int) -> Optional[Tuple[float, float]]:
        search_radius = self.cell_size * ANCHOR_SIZE
        y1 = max(0, approx_y - search_radius)
        y2 = min(image.shape[0], approx_y + search_radius)
        x1 = max(0, approx_x - search_radius)
        x2 = min(image.shape[1], approx_x + search_radius)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        import cv2
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary)
        if moments["m00"] == 0:
            return (float(approx_x), float(approx_y))

        cx = moments["m10"] / moments["m00"] + x1
        cy = moments["m01"] / moments["m00"] + y1
        return (cx, cy)

    def _normalize_colors(self, image: np.ndarray) -> np.ndarray:
        return image

    def reassemble(self, frames: dict) -> bytes:
        if not frames:
            return b""
        max_frame = max(frames.keys())
        parts = []
        for i in range(max_frame + 1):
            if i in frames:
                parts.append(frames[i].payload)
            else:
                parts.append(b"")
        return b"".join(parts)

    @staticmethod
    def extract_metadata(data: bytes) -> Tuple[str, bytes]:
        """Separate filename and file body from reassemble result."""
        if len(data) < 2:
            return ("", data)
        name_len = struct.unpack(">H", data[:2])[0]
        if len(data) < 2 + name_len:
            return ("", data)
        filename = data[2:2 + name_len].decode("utf-8", errors="replace")
        file_data = data[2 + name_len:]
        return (filename, file_data)
