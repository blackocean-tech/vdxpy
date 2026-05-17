"""Streaming reception reassembler.
Writes received payloads incrementally to disk, keeping memory usage at O(metadata).
Avoids OOM from `received_frames` dict with 5GB+ files."""
import os
import struct
import tempfile
from typing import Optional


class StreamingReassembler:
    """Writes frame payloads to temp file slots, then extracts metadata
    and copies to the final file on completion.

    Layout:
      slot N start offset = N * payload_capacity (sparse allocation)
      slot size = payload_capacity
      all frames except the last are exactly payload_capacity bytes
    """

    def __init__(self, payload_capacity: int, total_frames: int,
                 temp_dir: Optional[str] = None):
        self.payload_capacity = payload_capacity
        self.total_frames = total_frames
        self.temp_path = tempfile.NamedTemporaryFile(
            dir=temp_dir, delete=False, suffix=".qrtemp"
        ).name
        self._f = open(self.temp_path, "wb+")
        self._received = set()  # set of frame_num
        self._last_frame_len: Optional[int] = None

    def add_frame(self, frame_num: int, payload: bytes,
                  is_last: bool) -> bool:
        """Add a frame. Returns False if already received, True if newly added."""
        if frame_num in self._received:
            return False
        offset = frame_num * self.payload_capacity
        self._f.seek(offset)
        self._f.write(payload)
        self._received.add(frame_num)
        if is_last:
            self._last_frame_len = len(payload)
        return True

    @property
    def num_received(self) -> int:
        return len(self._received)

    def is_complete(self) -> bool:
        return len(self._received) >= self.total_frames

    def missing_frames(self):
        return set(range(self.total_frames)) - self._received

    def finalize_to(self, output_path: str) -> Optional[str]:
        """Extract metadata (filename) from temp file and write to final file.
        Returns: extracted filename (or None if invalid)."""
        K = self.total_frames
        last_len = self._last_frame_len or self.payload_capacity
        total_data_size = (K - 1) * self.payload_capacity + last_len

        self._f.flush()
        self._f.seek(0)
        # filename header: [2byte len][filename][file_data...]
        header = self._f.read(2)
        if len(header) < 2:
            return None
        name_len = struct.unpack(">H", header)[0]
        if name_len > 1024:
            # Invalid value, no metadata
            self._f.seek(0)
            with open(output_path, "wb") as out:
                self._copy_to(out, total_data_size)
            return None

        filename_bytes = self._f.read(name_len)
        filename = filename_bytes.decode("utf-8", errors="replace")
        # filedata length = total_data_size - 2 - name_len
        remaining = total_data_size - 2 - name_len

        with open(output_path, "wb") as out:
            self._copy_to(out, remaining)
        return filename

    def _copy_to(self, out_f, n_bytes: int):
        """Copy n_bytes from current temp file position to out_f"""
        remaining = n_bytes
        chunk_size = 4 * 1024 * 1024
        while remaining > 0:
            n = min(chunk_size, remaining)
            buf = self._f.read(n)
            if not buf:
                break
            out_f.write(buf)
            remaining -= len(buf)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
        try:
            os.unlink(self.temp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # Self-test
    import hashlib
    import os
    payload_cap = 100
    K = 50
    chunks = []
    fname = b"test.bin"
    src = b"\x00" * 2 + fname + os.urandom(K * payload_cap - len(fname) - 2 - 5)  # 5 extra bytes
    # Build: [2byte name_len=8][filename 8byte][rest]
    src = struct.pack(">H", len(fname)) + fname + os.urandom(K * payload_cap - len(fname) - 2 - 5)
    # K-1 chunks of payload_cap, last is 95
    chunks_data = [src[i*payload_cap:(i+1)*payload_cap] for i in range(K-1)]
    chunks_data.append(src[(K-1)*payload_cap:])

    r = StreamingReassembler(payload_cap, K)
    # Insert in random order
    import random
    order = list(range(K))
    random.shuffle(order)
    for i in order:
        r.add_frame(i, chunks_data[i], is_last=(i == K-1))
    assert r.is_complete()
    assert not r.missing_frames()

    out = "_st_test.bin"
    fn = r.finalize_to(out)
    r.close()
    assert fn == "test.bin", f"filename mismatch: {fn}"
    with open(out, "rb") as f:
        body = f.read()
    expected = src[2 + len(fname):]
    assert body == expected, f"body mismatch ({len(body)} vs {len(expected)})"
    os.unlink(out)
    print(f"PASS: K={K} frames roundtripped via temp file, filename='{fn}'")
