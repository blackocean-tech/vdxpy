"""Receiver - Reads frames from capture device and reconstructs the file"""
import argparse
import os
import struct
import sys
import time

from config import load_profile
from codec import ColorQRCodec, unwrap_zstd, ZSTD_MAGIC
from capture import create_capture


def _fec_can_decode(received_frames, K, M, G):
    """Check if all groups have received at least k_g frames (FEC decodability check)"""
    n_groups = (K + G - 1) // G
    for g in range(n_groups):
        k_g = min(G, K - g * G)
        # group data: [g*G, g*G + k_g), parity: [K + g*M, K + (g+1)*M)
        cnt = 0
        for i in range(k_g):
            if g * G + i in received_frames:
                cnt += 1
        for i in range(M):
            if K + g * M + i in received_frames:
                cnt += 1
        if cnt < k_g:
            return False
    return True


def _fec_decode_received(received_frames, K, M, G, payload_size):
    """Decode received data per FEC group and recover K data payloads.
    received_frames: dict{frame_num: FrameData}
    Returns: list of K bytes (None on failure).
    """
    from frame_fec import decode_with_fec
    n_groups = (K + G - 1) // G
    result = []
    for g in range(n_groups):
        k_g = min(G, K - g * G)
        # group data frame_num: [g*G, g*G + k_g)
        # group parity frame_num: [K + g*M, K + (g+1)*M)
        group_received = {}
        for i in range(k_g):
            global_idx = g * G + i
            fd = received_frames.get(global_idx)
            if fd is not None:
                # data frame payload at original length (payload_size except for last)
                p = fd.payload
                if len(p) < payload_size:
                    p = p + b"\x00" * (payload_size - len(p))
                group_received[i] = p
        for i in range(M):
            global_idx = K + g * M + i
            fd = received_frames.get(global_idx)
            if fd is not None:
                # parity payload is always payload_size length (aligned by encode_with_fec)
                p = fd.payload
                if len(p) < payload_size:
                    p = p + b"\x00" * (payload_size - len(p))
                group_received[k_g + i] = p
        if len(group_received) < k_g:
            return None  # Not enough frames
        recovered = decode_with_fec(group_received, k_g, M, payload_size)
        if recovered is None:
            return None
        result.extend(recovered)
    return result


def main():
    parser = argparse.ArgumentParser(description="QR File Transfer - Receiver")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for reconstructed file (uses sender's filename if omitted)")
    parser.add_argument("--device", "-d", type=int, default=0,
                        help="Capture device number (default: 0)")
    parser.add_argument("--profile", default="cvhduvc2",
                        help="Profile name (default: cvhduvc2)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout in seconds when no new frames received (default: 60)")
    parser.add_argument("--video", default=None,
                        help="Play from recorded file (instead of device)")
    parser.add_argument("--config", default=None,
                        help="Path to profile config file")
    parser.add_argument("--no-threaded", action="store_true",
                        help="Disable threaded capture (real device capture only)")
    parser.add_argument("--progress-interval", type=float, default=0.5,
                        help="Minimum interval in seconds for progress display (default: 0.5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress display to minimize CPU/IO load")
    parser.add_argument("--streaming", action="store_true",
                        help="Write payload to disk immediately (for large files, reduces memory usage)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers for parallel full RS decode (#3, default: 1)")
    args = parser.parse_args()

    config = load_profile(args.profile, args.config)
    config.capture.device_index = args.device

    codec = ColorQRCodec(config)

    capture_kwargs = {}
    if args.video:
        capture_kwargs["video_path"] = args.video
    elif not args.no_threaded:
        capture_kwargs["threaded"] = True  # Countermeasure 1
    capture = create_capture(config, **capture_kwargs)

    print(f"[Receiver] Profile: {config.name}")
    print(f"[Receiver] Resolution: {config.capture.resolution}")
    print(f"[Receiver] Capture: {type(capture).__name__}")
    print(f"[Receiver] Output: {args.output}")

    received_frames = {}  # Used only when streaming=False
    streaming = None      # StreamingReassembler when streaming=True
    if args.streaming:
        from streaming_reassembler import StreamingReassembler

    # #3: Multi-threaded decode (effective on 2-PC setup, negligible on single-PC)
    pending_fns: set = set()
    in_flight: list = []
    executor = None
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=args.workers)
        print(f"[Receiver] decode workers: {args.workers}")

    # FEC settings: must match sender (read from profile)
    fec_M = config.codec.fec_M_per_group
    fec_G = config.codec.fec_group_size
    fec_K = None  # Computed from first frame header
    if fec_M > 0:
        print(f"[Receiver] FEC mode: M={fec_M}/group, group_size={fec_G}")

    def fec_compute_K(total, M, G):
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

    total_frames = None
    last_receive_time = 0.0
    frame_count = 0
    error_count = 0
    skipped_known = 0
    skipped_by_fp = 0
    last_progress_time = 0.0
    known_fingerprints: dict = {}

    def n_received():
        return streaming.num_received if streaming else len(received_frames)

    last_receive_time = time.monotonic()

    with capture:
        print("[Receiver] Capturing frames... (Ctrl+C to stop)")
        try:
            while True:
                raw_frame = capture.read_frame()
                # Q3: Use time.monotonic() consistently (high precision and monotonic on Windows)
                now = time.monotonic()
                # Timeout if no new unique frame arrives within specified time
                # (coupon collector termination when frames keep coming but no new unique ones)
                if last_receive_time > 0 and now - last_receive_time > args.timeout:
                    print(f"\n[Receiver] No new unique frame for {args.timeout}s, stopping.")
                    break
                if raw_frame is None:
                    continue

                frame_count += 1

                # Countermeasure 7 (fingerprint early skip): sample only 32 cells to detect known frames.
                fp = codec.compute_fingerprint(raw_frame)
                if fp and fp in known_fingerprints:
                    skipped_by_fp += 1
                else:
                    # Countermeasure 3: fast decode of header only
                    hdr = codec.decode_frame_header(raw_frame)
                    if hdr is None:
                        error_count += 1
                    else:
                        fn, tot, raw_bytes = hdr
                        if total_frames is None:
                            total_frames = tot
                            print(f"\n[Receiver] Total frames expected: {total_frames}")
                            if fec_M > 0:
                                fec_K = fec_compute_K(total_frames, fec_M, fec_G)
                                print(f"[Receiver] FEC: K(data)={fec_K}, "
                                      f"parity={total_frames-fec_K}")

                        # Initialize streaming mode (on first frame arrival)
                        if args.streaming and streaming is None:
                            streaming = StreamingReassembler(
                                codec.payload_capacity, total_frames
                            )

                        is_known = (
                            (fn in streaming._received) if streaming
                            else (fn in received_frames)
                        ) or (fn in pending_fns)
                        if is_known:
                            if fp:
                                known_fingerprints[fp] = fn
                            skipped_known += 1
                        elif executor is not None:
                            # #3: Submit full decode to worker
                            pending_fns.add(fn)
                            if raw_bytes:
                                fut = executor.submit(
                                    codec.decode_frame_from_bytes, raw_bytes
                                )
                            else:
                                fut = executor.submit(
                                    codec.decode_frame, raw_frame
                                )
                            in_flight.append((fut, fn, fp))
                        else:
                            if raw_bytes:
                                frame_data = codec.decode_frame_from_bytes(raw_bytes)
                            else:
                                frame_data = codec.decode_frame(raw_frame)
                            if frame_data is None:
                                error_count += 1
                            else:
                                if streaming is not None:
                                    streaming.add_frame(
                                        frame_data.frame_number,
                                        frame_data.payload,
                                        frame_data.is_last
                                    )
                                else:
                                    received_frames[frame_data.frame_number] = frame_data
                                if fp:
                                    known_fingerprints[fp] = frame_data.frame_number
                                last_receive_time = now

                # #3: Collect completed workers
                if executor is not None and in_flight:
                    still = []
                    for fut, fn_p, fp_p in in_flight:
                        if not fut.done():
                            still.append((fut, fn_p, fp_p))
                            continue
                        try:
                            fd = fut.result()
                        except Exception:
                            fd = None
                        pending_fns.discard(fn_p)
                        if fd is None:
                            error_count += 1
                        else:
                            if streaming is not None:
                                streaming.add_frame(
                                    fd.frame_number, fd.payload, fd.is_last
                                )
                            else:
                                received_frames[fd.frame_number] = fd
                            if fp_p:
                                known_fingerprints[fp_p] = fd.frame_number
                            last_receive_time = now
                    in_flight = still

                # Q4: Progress display suppressible with --quiet, default interval expanded to 0.5s
                if not args.quiet and now - last_progress_time >= args.progress_interval:
                    sys.stdout.write(
                        f"\r[Receiver] Captured: {frame_count} | "
                        f"Decoded: {n_received()}/{total_frames or '?'} | "
                        f"Skipped: {skipped_known}+{skipped_by_fp}fp | "
                        f"Errors: {error_count}"
                    )
                    sys.stdout.flush()
                    last_progress_time = now

                if total_frames and n_received() >= total_frames:
                    print("\n[Receiver] All frames received!")
                    break

                # FEC mode: early termination when all groups have minimum required frames
                if fec_K is not None and not args.streaming and frame_count % 50 == 0:
                    if _fec_can_decode(received_frames, fec_K, fec_M, fec_G):
                        print(f"\n[Receiver] FEC: all groups recoverable ({n_received()}/{total_frames})")
                        break

        except KeyboardInterrupt:
            print("\n[Receiver] Interrupted by user.")

    # #3: Drain remaining in_flight futures
    if executor is not None:
        for fut, fn_p, fp_p in in_flight:
            try:
                fd = fut.result(timeout=2.0)
            except Exception:
                fd = None
            pending_fns.discard(fn_p)
            if fd is not None:
                if streaming is not None:
                    streaming.add_frame(fd.frame_number, fd.payload, fd.is_last)
                elif fd.frame_number not in received_frames:
                    received_frames[fd.frame_number] = fd
        in_flight.clear()
        executor.shutdown(wait=False)

    # Display ThreadedCapture statistics
    if hasattr(capture, "stats"):
        s = capture.stats
        print(f"[Receiver] Capture stats: total={s['captured_total']}, "
              f"dropped(unread)={s['dropped']}")

    if n_received() == 0:
        print("[Receiver] No frames decoded. Exiting.")
        sys.exit(1)

    if total_frames and n_received() < total_frames:
        if streaming is not None:
            missing = streaming.missing_frames()
        else:
            missing = set(range(total_frames)) - set(received_frames.keys())
        if len(missing) <= 50:
            print(f"[Receiver] WARNING: Missing frames: {sorted(missing)}")
        else:
            print(f"[Receiver] WARNING: Missing {len(missing)} frames")

    print("[Receiver] Reassembling file...")

    if streaming is not None:
        # Streaming: write final file directly from temp file
        if args.output:
            output_path = args.output
        else:
            output_path = "received_file"  # Temporary default, renamed after filename extraction
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Finalize while extracting filename
        filename = streaming.finalize_to(output_path)
        if filename and not args.output:
            new_path = filename
            os.makedirs(os.path.dirname(os.path.abspath(new_path)) or ".",
                        exist_ok=True)
            os.replace(output_path, new_path)
            output_path = new_path
            print(f"[Receiver] Filename from sender: {filename}")
        # B1: Detect leading magic in streaming path and decompress zstd (streaming decompression)
        with open(output_path, "rb") as _hf:
            head = _hf.read(len(ZSTD_MAGIC))
        if head == ZSTD_MAGIC:
            import zstandard as zstd
            dec_path = output_path + ".dec"
            comp_size = os.path.getsize(output_path)
            with open(output_path, "rb") as src, open(dec_path, "wb") as dst:
                src.read(ZSTD_MAGIC.__len__())  # skip magic
                _orig_size = struct.unpack(">Q", src.read(8))[0]
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(src, dst)
            os.replace(dec_path, output_path)
            print(f"[Receiver] zstd decompressed (streaming): "
                  f"{comp_size:,} -> {os.path.getsize(output_path):,} bytes")
        out_size = os.path.getsize(output_path)
        streaming.close()
    else:
        # Legacy: reassemble in memory
        if fec_M > 0 and fec_K is not None:
            # FEC path: check if K data frames are complete, supplement with parity if not
            print(f"[Receiver] FEC decode (K={fec_K}, M={fec_M}/group, G={fec_G})...")
            data_payloads = _fec_decode_received(
                received_frames, fec_K, fec_M, fec_G, codec.payload_capacity
            )
            if data_payloads is None:
                print("[Receiver] FEC decode FAILED (insufficient frames)")
                missing_data = [i for i in range(fec_K) if i not in received_frames]
                missing_data_count = sum(
                    1 for i in range(fec_K) if i not in received_frames
                )
                missing_parity_count = sum(
                    1 for fn in range(fec_K, total_frames)
                    if fn not in received_frames
                )
                print(f"  missing data={missing_data_count}/{fec_K}, "
                      f"missing parity={missing_parity_count}/{total_frames-fec_K}")
                sys.exit(1)
            # data_payloads contains K chunks (padded to uniform length)
            # The last chunk may be shorter in the original file -> get actual length
            # of the last chunk from received_frames for proper reconstruction
            if (fec_K - 1) in received_frames:
                last_len = len(received_frames[fec_K - 1].payload)
                data_payloads[fec_K - 1] = data_payloads[fec_K - 1][:last_len]
            raw_data = b"".join(data_payloads)
            filename, data = codec.extract_metadata(raw_data)
        else:
            raw_data = codec.reassemble(received_frames)
            filename, data = codec.extract_metadata(raw_data)
        # B1: Decompress if payload was pre-compressed with zstd
        if data.startswith(ZSTD_MAGIC):
            comp_size = len(data)
            data = unwrap_zstd(data)
            print(f"[Receiver] zstd decompressed: {comp_size:,} -> {len(data):,} bytes "
                  f"(ratio {comp_size/len(data):.3f}x)")
        if args.output:
            output_path = args.output
        elif filename:
            output_path = filename
            print(f"[Receiver] Filename from sender: {filename}")
        else:
            output_path = "received_file"
            print("[Receiver] WARNING: No filename in data, using default.")
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        out_size = len(data)

    print(f"[Receiver] File saved: {output_path} ({out_size} bytes)")
    print(f"[Receiver] Stats: {frame_count} frames captured, "
          f"{n_received()} decoded, {skipped_known} known-skipped, "
          f"{skipped_by_fp} fp-skipped, {error_count} errors")


if __name__ == "__main__":
    main()
