"""Integrated benchmark: runs sender (thread) + receiver in a single process.
The sender runs in a background thread to avoid pygame focus-loss issues.

Usage:
  python bench_integrated.py --device 2 --screen 1 --sizes 5000,50000,500000,5000000
"""
import argparse
import hashlib
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np

from config import load_profile
from codec import ColorQRCodec, _HAS_CREEDSOLO, ZSTD_MAGIC, unwrap_zstd
from capture import create_capture


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fec_compute_K(total, M, G):
    """Solve K + M * ceil(K/G) = total for K"""
    if M <= 0:
        return total
    for ng in range(1, total // M + 2):
        K = total - M * ng
        if K <= 0:
            break
        if (K + G - 1) // G == ng:
            return K
    return None


def make_test_data(size: int, pattern: str) -> bytes:
    """Generate benchmark data. pattern:
       - random: fully random (zstd 1.0x)
       - text:   ASCII text-like high-redundancy data (zstd ~3-10x)
       - log:    log-like (timestamp + repeated structured messages, zstd ~5-20x)
    """
    if pattern == "random":
        return os.urandom(size)
    if pattern == "text":
        sentence = (
            b"The quick brown fox jumps over the lazy dog. "
            b"Pack my box with five dozen liquor jugs. "
            b"How vexingly quick daft zebras jump! "
            b"Sphinx of black quartz, judge my vow.\n"
        )
        rep = (size // len(sentence)) + 1
        return (sentence * rep)[:size]
    if pattern == "log":
        line_tmpl = (
            b"2026-05-09T12:34:56.789Z [INFO ] worker[%05d] "
            b"request_id=abc-123-def-456 user=alice "
            b"path=/api/v1/items status=200 latency_ms=42 bytes=1024\n"
        )
        out = bytearray()
        i = 0
        while len(out) < size:
            out += line_tmpl % (i & 0xFFFFF)
            i += 1
        return bytes(out[:size])
    raise ValueError(f"unknown data pattern: {pattern}")


class SenderThread(threading.Thread):
    """OpenCV imshow-based sender thread.
    Since imshow must be called from the main thread, frames are pumped from main.
    """
    def __init__(self, cellgrid_path, config, codec, screen=1, shuffle_seed=None):
        super().__init__(daemon=True)
        self.cellgrid_path = cellgrid_path
        self.config = config
        self.codec = codec
        self.screen = screen
        self.running = False
        self.grids = []
        self.palette = codec._ref_bgr.astype(np.uint8)
        self._stop_event = threading.Event()
        # #2: pseudo-random frame ordering (reproducible with fixed seed)
        self.shuffle_seed = shuffle_seed
        self._order = None  # None means sequential order

    def load(self):
        with open(self.cellgrid_path, "rb") as f:
            rows, cols = struct.unpack(">HH", f.read(4))
            frame_size = rows * cols
            while True:
                data = f.read(frame_size)
                if len(data) < frame_size:
                    break
                self.grids.append(
                    np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
                )
        # #2: shuffle order if seed provided
        if self.shuffle_seed is not None:
            import random
            order = list(range(len(self.grids)))
            random.Random(self.shuffle_seed).shuffle(order)
            self._order = order
        return len(self.grids)

    def make_image(self, idx):
        n = len(self.grids)
        if self._order is not None:
            grid = self.grids[self._order[idx % n]]
        else:
            grid = self.grids[idx % n]
        bgr_small = self.palette[grid]
        w, h = self.config.display.resolution
        return cv2.resize(bgr_small, (w, h), interpolation=cv2.INTER_NEAREST)

    def stop(self):
        self._stop_event.set()


def run_receiver(config, codec, device, timeout, output_path, quiet=True,
                 workers=1, lt_mode=False):
    """Receiver logic extracted as a function (simplified version).
    workers > 1 parallelizes full RS decode via ThreadPoolExecutor (#3).
    lt_mode=True enables LT code reception (recovers from any K+epsilon blocks, #4).
    """
    config.capture.device_index = device
    capture = create_capture(config, threaded=True)

    received_frames = {}
    total_frames = None
    last_receive_time = time.monotonic()
    frame_count = 0
    error_count = 0
    skipped_known = 0
    skipped_by_fp = 0
    known_fingerprints = {}
    pending_fns = set()  # frame_nums submitted to decode worker, awaiting completion
    in_flight = []       # list of [(future, frame_num, fingerprint)]

    # #4: LT decoder initialization
    lt_decoder = None
    lt_consumed_seeds: set = set()
    lt_done = False
    if lt_mode:
        import lt.decode as ltd
        lt_decoder = ltd.LtDecoder()

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=workers)
    else:
        executor = None

    def drain_completed():
        """Collect completed futures into received_frames."""
        nonlocal error_count, last_receive_time
        still = []
        for fut, fn, fp in in_flight:
            if not fut.done():
                still.append((fut, fn, fp))
                continue
            try:
                fd = fut.result()
            except Exception:
                fd = None
            pending_fns.discard(fn)
            if fd is None:
                error_count += 1
            else:
                if fd.frame_number not in received_frames:
                    received_frames[fd.frame_number] = fd
                    last_receive_time = time.monotonic()
                if fp:
                    known_fingerprints[fp] = fd.frame_number
        in_flight[:] = still

    with capture:
        while True:
            raw_frame = capture.read_frame()
            now = time.monotonic()
            if executor:
                drain_completed()
            if last_receive_time > 0 and now - last_receive_time > timeout:
                break
            if raw_frame is None:
                continue
            frame_count += 1

            fp = codec.compute_fingerprint(raw_frame)
            if fp and fp in known_fingerprints:
                skipped_by_fp += 1
            else:
                hdr = codec.decode_frame_header(raw_frame)
                if hdr is None:
                    error_count += 1
                else:
                    fn, tot, raw_bytes = hdr
                    if total_frames is None:
                        total_frames = tot

                    if lt_mode:
                        if fn in lt_consumed_seeds:
                            if fp:
                                known_fingerprints[fp] = fn
                            skipped_known += 1
                        else:
                            if raw_bytes:
                                frame_data = codec.decode_frame_from_bytes(raw_bytes)
                            else:
                                frame_data = codec.decode_frame(raw_frame)
                            if frame_data is None:
                                error_count += 1
                            else:
                                import lt.decode as ltd
                                try:
                                    lt_decoder.consume_block(
                                        ltd.block_from_bytes(frame_data.payload)
                                    )
                                    lt_consumed_seeds.add(fn)
                                    if fp:
                                        known_fingerprints[fp] = fn
                                    last_receive_time = now
                                    if lt_decoder.is_done():
                                        lt_done = True
                                except Exception:
                                    error_count += 1
                    elif fn in received_frames or fn in pending_fns:
                        # #5: fast path - skip without full sampling
                        if fp:
                            known_fingerprints[fp] = fn
                        skipped_known += 1
                    elif executor is not None:
                        pending_fns.add(fn)
                        # full decode from image (raw_bytes is empty in fast path)
                        if raw_bytes:
                            fut = executor.submit(codec.decode_frame_from_bytes, raw_bytes)
                        else:
                            fut = executor.submit(codec.decode_frame, raw_frame)
                        in_flight.append((fut, fn, fp))
                    else:
                        if raw_bytes:
                            frame_data = codec.decode_frame_from_bytes(raw_bytes)
                        else:
                            frame_data = codec.decode_frame(raw_frame)
                        if frame_data is None:
                            error_count += 1
                        else:
                            received_frames[frame_data.frame_number] = frame_data
                            if fp:
                                known_fingerprints[fp] = frame_data.frame_number
                            last_receive_time = now

            if lt_mode:
                if lt_done:
                    break
            elif total_frames and len(received_frames) >= total_frames:
                break

    # Drain remaining in_flight futures from executor
    if executor is not None:
        # Wait for remaining futures to complete (with timeout)
        for fut, fn, fp in in_flight:
            try:
                fd = fut.result(timeout=2.0)
            except Exception:
                fd = None
            pending_fns.discard(fn)
            if fd is not None and fd.frame_number not in received_frames:
                received_frames[fd.frame_number] = fd
        in_flight.clear()
        executor.shutdown(wait=False)

    decoded_count = (
        len(lt_consumed_seeds) if lt_mode else len(received_frames)
    )
    stats = {
        "frame_count": frame_count,
        "decoded": decoded_count,
        "total": total_frames,
        "errors": error_count,
        "skipped_known": skipped_known,
        "skipped_fp": skipped_by_fp,
        "workers": workers,
        "lt_mode": lt_mode,
    }

    if lt_mode:
        if not (lt_decoder and lt_decoder.is_done()):
            return None, stats
        import io as _io
        out_buf = _io.BytesIO()
        lt_decoder.stream_dump(out_buf)
        raw_data = out_buf.getvalue()
    else:
        if len(received_frames) == 0:
            return None, stats
        fec_M = config.codec.fec_M_per_group
        fec_G = config.codec.fec_group_size
        if fec_M > 0 and total_frames is not None:
            from receiver import _fec_decode_received
            fec_K = _fec_compute_K(total_frames, fec_M, fec_G)
            data_payloads = _fec_decode_received(
                received_frames, fec_K, fec_M, fec_G,
                codec.payload_capacity
            )
            if data_payloads is None:
                return None, stats
            if (fec_K - 1) in received_frames:
                last_len = len(received_frames[fec_K - 1].payload)
                data_payloads[fec_K - 1] = data_payloads[fec_K - 1][:last_len]
            raw_data = b"".join(data_payloads)
            stats["fec_K"] = fec_K
        else:
            raw_data = codec.reassemble(received_frames)

    filename, data = codec.extract_metadata(raw_data)
    if data.startswith(ZSTD_MAGIC):
        comp_size = len(data)
        data = unwrap_zstd(data)
        stats["zstd_comp"] = comp_size
        stats["zstd_orig"] = len(data)
    with open(output_path, "wb") as f:
        f.write(data)
    return len(data), stats


def run_test(size, profile_name, device, screen, timeout, work_dir,
             compress=False, compress_level=19, data_pattern="random",
             workers=1, lt_mode=False, lt_oversampling=1.5,
             shuffle_seed=None, display_fps=None):
    label = f"{size/1000:.0f}KB" if size < 1_000_000 else f"{size/1_000_000:.0f}MB"
    tag = label
    if compress:
        tag += f"_zstd{compress_level}"
    if data_pattern != "random":
        tag += f"_{data_pattern}"
    print(f"\n{'='*60}")
    print(f"  Test: {tag} ({size:,} bytes, pattern={data_pattern}, "
          f"compress={'zstd L'+str(compress_level) if compress else 'off'})")
    print(f"{'='*60}")

    src = os.path.join(work_dir, f"bench_src_{tag}.bin")
    cellgrid = os.path.join(work_dir, f"bench_{tag}.qrcells")
    out = os.path.join(work_dir, f"bench_out_{tag}.bin")

    for f in [src, cellgrid, out]:
        if os.path.exists(f):
            os.remove(f)

    config = load_profile(profile_name)
    codec = ColorQRCodec(config)

    # 1. Generate test file
    with open(src, "wb") as f:
        f.write(make_test_data(size, data_pattern))
    src_hash = sha256(src)

    # 2. Prerender (cellgrid) - run in-process
    if lt_mode:
        chunks = codec.prepare_chunks_lt(
            src, oversampling=lt_oversampling,
            compress=compress, compress_level=compress_level,
        )
    else:
        chunks = codec.prepare_chunks(src, compress=compress, compress_level=compress_level)
    total = len(chunks)
    rows = config.display.resolution[1] // codec.cell_size
    cols = config.display.resolution[0] // codec.cell_size
    with open(cellgrid, "wb") as f:
        f.write(struct.pack(">HH", rows, cols))
        for frame_num, payload, total_frames, is_last in chunks:
            cells = codec.encode_chunk_to_cells(frame_num, payload, total_frames, is_last)
            grid = np.array(cells, dtype=np.uint8).reshape(rows, cols)
            f.write(grid.tobytes())
    print(f"  Prerender: {total} frames, cellgrid={os.path.getsize(cellgrid)} bytes")

    # 3. Prepare sender
    sender = SenderThread(cellgrid, config, codec, screen, shuffle_seed=shuffle_seed)
    n_frames = sender.load()

    # 4. Display sender in OpenCV window + run receiver concurrently
    win_name = "QR_Bench_Sender"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if screen == 1:
        cv2.moveWindow(win_name, 1920, 0)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the first frame and wait for sender to stabilize
    img = sender.make_image(0)
    cv2.imshow(win_name, img)
    cv2.waitKey(1)
    time.sleep(2.0)

    # Run receiver in a background thread
    recv_result = [None, None]
    t_start = [0.0]
    t_end = [0.0]

    def recv_thread():
        t_start[0] = time.perf_counter()
        data_len, stats = run_receiver(config, codec, device, timeout, out,
                                       quiet=True, workers=workers,
                                       lt_mode=lt_mode)
        t_end[0] = time.perf_counter()
        recv_result[0] = data_len
        recv_result[1] = stats

    rt = threading.Thread(target=recv_thread)
    rt.start()

    # Pump sender display in the main thread
    idx = 0
    fps = display_fps if display_fps else config.display.fps
    while rt.is_alive():
        img = sender.make_image(idx)
        cv2.imshow(win_name, img)
        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key == 27:
            break
        idx = (idx + 1) % n_frames

    rt.join(timeout=5)
    cv2.destroyWindow(win_name)
    cv2.waitKey(1)

    # 5. Results
    t_recv = t_end[0] - t_start[0] if t_end[0] > 0 else 0
    stats = recv_result[1] or {}

    if recv_result[0] is None or not os.path.exists(out):
        print(f"  FAILED: no data received")
        print(f"  Stats: {stats}")
        for f in [src, cellgrid, out]:
            if os.path.exists(f):
                os.remove(f)
        return None

    out_hash = sha256(out)
    match = src_hash == out_hash
    throughput = size / t_recv if t_recv > 0 else 0

    print(f"  Receive time: {t_recv:.1f}s")
    print(f"  Throughput: {throughput/1000:.1f} KB/s")
    print(f"  SHA256 match: {'PASS' if match else 'FAIL'}")
    print(f"  Stats: captured={stats.get('frame_count',0)}, "
          f"decoded={stats.get('decoded',0)}/{stats.get('total','?')}, "
          f"errors={stats.get('errors',0)}")

    for f in [src, cellgrid, out]:
        if os.path.exists(f):
            os.remove(f)

    return {
        "size": size,
        "label": label,
        "tag": tag,
        "compress": compress,
        "compress_level": compress_level if compress else None,
        "data_pattern": data_pattern,
        "time_s": round(t_recv, 1),
        "throughput_kBs": round(throughput / 1000, 1),
        "sha256_match": match,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Integrated Transfer Benchmark")
    parser.add_argument("--device", "-d", type=int, default=2)
    parser.add_argument("--screen", type=int, default=1)
    parser.add_argument("--profile", default="hdmi_plus")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sizes", default="5000,50000,500000,5000000",
                        help="comma-separated test sizes in bytes")
    parser.add_argument("--compress", action="store_true",
                        help="pre-compress payload with zstd")
    parser.add_argument("--compress-level", type=int, default=19,
                        help="zstd compression level 1-22 (default: 19)")
    parser.add_argument("--data-pattern", choices=["random", "text", "log"],
                        default="random",
                        help="benchmark data pattern (default: random)")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of receiver decode workers (#3, default: 1=single thread)")
    parser.add_argument("--lt", action="store_true",
                        help="LT (fountain) code mode (#4, eliminates coupon collector tail)")
    parser.add_argument("--lt-oversampling", type=float, default=1.5,
                        help="LT encoded block count = K * oversampling (default: 1.5)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="seed to pseudo-randomize sender playback order (#2)")
    parser.add_argument("--display-fps", type=int, default=None,
                        help="override sender display fps (default: profile's display.fps)")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    work_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"creedsolo active: {_HAS_CREEDSOLO}")
    print(f"Profile: {args.profile}")
    print(f"Device: {args.device}, Screen: {args.screen}")
    print(f"Data pattern: {args.data_pattern}, Compress: "
          f"{'zstd L'+str(args.compress_level) if args.compress else 'off'}, "
          f"Workers: {args.workers}, LT: "
          f"{'ON x'+str(args.lt_oversampling) if args.lt else 'off'}, "
          f"Shuffle seed: {args.shuffle_seed}")

    results = []
    for size in sizes:
        timeout = max(args.timeout, size / 500 + 30)
        result = run_test(size, args.profile, args.device, args.screen,
                          timeout, work_dir,
                          compress=args.compress,
                          compress_level=args.compress_level,
                          data_pattern=args.data_pattern,
                          workers=args.workers,
                          lt_mode=args.lt,
                          lt_oversampling=args.lt_oversampling,
                          shuffle_seed=args.shuffle_seed,
                          display_fps=args.display_fps)
        if result:
            results.append(result)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  creedsolo: {_HAS_CREEDSOLO}")
    print(f"{'Size':>10} {'Time':>8} {'Throughput':>12} {'SHA256':>8}")
    print(f"{'-'*10} {'-'*8} {'-'*12} {'-'*8}")
    for r in results:
        print(f"{r['label']:>10} {r['time_s']:>7.1f}s {r['throughput_kBs']:>10.1f} KB/s "
              f"{'PASS' if r['sha256_match'] else 'FAIL':>8}")

    baseline = {"5KB": 1.7, "50KB": 15, "500KB": 51, "5MB": 69}
    print(f"\n  Comparison (before creedsolo / 1-PC):")
    for r in results:
        bl = baseline.get(r["label"])
        if bl:
            ratio = r["throughput_kBs"] / bl if bl > 0 else 0
            print(f"    {r['label']}: {bl:.0f} -> {r['throughput_kBs']:.1f} KB/s ({ratio:.1f}x)")

    import json
    out_json = os.path.join(work_dir, "bench_transfer_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "creedsolo": _HAS_CREEDSOLO,
            "profile": args.profile,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_json}")


if __name__ == "__main__":
    main()
