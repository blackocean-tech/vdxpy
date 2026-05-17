"""Tool for pre-encoding files into QR frame video.

Benefits:
- Eliminates RS encode overhead during transmission; playback is always at stable display.fps
- Reusable when sending the same file multiple times
- Minimizes CPU contention during transmission (effective for single-PC receiver loop)
- Allows aggressive cell_size reduction without first-pass delay

Output formats:
- video mode (--codec ffv1 etc.): video file (.mkv) via cv2.VideoWriter
- cellgrid mode (--cellgrid): intermediate representation of cell values only (.qrcells)
  Much smaller than video (~1/100 of 1080p FFV1). Playback is palette+resize only.

Examples:
    # Video (FFV1 lossless):
    python prerender.py file.bin --profile hdmi_plus -o file.qrvid.mkv

    # Cell grid (recommended):
    python prerender.py file.bin --profile hdmi_plus --cellgrid -o file.qrcells

    # Playback:
    python sender.py --video file.qrvid.mkv --profile hdmi_plus --loop --screen 1
    python sender.py --cellgrid file.qrcells --profile hdmi_plus --loop --screen 1
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from config import load_profile
from codec import ColorQRCodec


# (fourcc, default ext, description)
CODEC_OPTIONS = {
    "ffv1":  ("FFV1", ".mkv", "Lossless FFV1. Zero color degradation, large size"),
    "huffyuv": ("HFYU", ".avi", "Lossless HuffYUV. Faster than FFV1 but large size"),
    "raw":   ("RGBA", ".avi", "Completely uncompressed RGBA. Largest size"),
    "mjpeg": ("MJPG", ".avi", "MJPEG light compression. Some degradation but much smaller"),
}


def main():
    parser = argparse.ArgumentParser(
        description="QR File Transfer - Pre-render encoder"
    )
    parser.add_argument("file", help="path to the file to transmit")
    parser.add_argument("--profile", default="hdmi_plus")
    parser.add_argument("--output", "-o", required=True,
                        help="output file")
    parser.add_argument("--codec", default="ffv1",
                        choices=list(CODEC_OPTIONS.keys()),
                        help="video codec (default: ffv1=lossless)")
    parser.add_argument("--cellgrid", action="store_true",
                        help="output as cell grid intermediate format (much smaller and faster than video)")
    parser.add_argument("--compress", action="store_true",
                        help="pre-compress payload with zstd (effective for text etc.)")
    parser.add_argument("--compress-level", type=int, default=19,
                        help="zstd compression level 1-22 (default: 19)")
    parser.add_argument("--config", default=None,
                        help="path to profile configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: input file not found: {args.file}")
        sys.exit(2)

    config = load_profile(args.profile, args.config)
    codec = ColorQRCodec(config)

    file_size = os.path.getsize(args.file)
    print(f"[Prerender] Profile: {config.name}")
    print(f"[Prerender] Input: {args.file} ({file_size:,} bytes)")
    print(f"[Prerender] Cell size: {config.codec.cell_size}px, "
          f"Colors: {config.codec.num_colors}")
    print(f"[Prerender] Payload/frame: {codec.payload_capacity} bytes")

    # Prepare chunks
    if args.compress:
        print(f"[Prerender] Compress: zstd level={args.compress_level}")
    chunks = codec.prepare_chunks(
        args.file, compress=args.compress, compress_level=args.compress_level
    )
    total = len(chunks)
    print(f"[Prerender] Total frames: {total}")

    if args.cellgrid:
        run_cellgrid_mode(args, config, codec, chunks, total, file_size)
    else:
        run_video_mode(args, config, codec, chunks, total, file_size)


def run_cellgrid_mode(args, config, codec, chunks, total, file_size):
    """Cell grid mode: saves up to RS encode + cell quantize stage.
    Each frame is grid_rows x grid_cols bytes (e.g., 32400 bytes = 32KB for hdmi_plus).
    Unlike video, this is byte-exact lossless, and file size is ~1/100."""
    out_path = args.output
    print(f"[Prerender] Output: {out_path} (cellgrid mode)")
    gr, gc = codec.grid_rows, codec.grid_cols
    print(f"[Prerender] Grid: {gr}×{gc} = {gr*gc} bytes/frame")

    # File header: magic(8) + grid_rows(2) + grid_cols(2) + total_frames(4) = 16 bytes
    import struct
    MAGIC = b"QRCELLS\x01"
    t_start = time.perf_counter()
    last_log = t_start
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">HHI", gr, gc, total))
        for info in chunks:
            grid = codec.encode_chunk_to_cells(*info)
            f.write(grid.tobytes())
            idx = info[0]
            now = time.perf_counter()
            if now - last_log >= 1.0:
                elapsed = now - t_start
                rate = (idx + 1) / elapsed
                eta = (total - idx - 1) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r[Prerender] {idx+1}/{total} "
                    f"({elapsed:.1f}s, {rate:.1f} fps, ETA {eta:.0f}s)"
                )
                sys.stdout.flush()
                last_log = now

    elapsed = time.perf_counter() - t_start
    out_size = os.path.getsize(out_path)
    print(f"\n[Prerender] Done in {elapsed:.1f}s ({total/elapsed:.1f} fps)")
    print(f"[Prerender] Output: {out_size:,} bytes "
          f"(ratio {out_size/file_size:.2f}x source)")
    print(f"\nPlayback: python sender.py --cellgrid {out_path} "
          f"--profile {args.profile} --loop --screen 1")


def run_video_mode(args, config, codec, chunks, total, file_size):
    """Video mode: saves completed images to a video file"""
    fourcc_str, default_ext, desc = CODEC_OPTIONS[args.codec]
    if not args.output.endswith(default_ext):
        print(f"WARN: codec {args.codec} recommends {default_ext} extension "
              f"(specified: {args.output})")
    w, h = config.display.resolution
    fps = config.display.fps
    print(f"[Prerender] Output: {args.output} (codec={args.codec}, {desc})")

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"ERROR: cv2.VideoWriter failed to open {args.output} "
              f"(codec={fourcc_str})")
        sys.exit(1)

    print("[Prerender] Encoding...")
    t_start = time.perf_counter()
    last_log = t_start
    for info in chunks:
        frame = codec.encode_chunk(*info)
        writer.write(frame)
        idx = info[0]
        now = time.perf_counter()
        if now - last_log >= 1.0:
            elapsed = now - t_start
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r[Prerender] {idx+1}/{total} "
                f"({elapsed:.1f}s, {rate:.1f} fps, ETA {eta:.0f}s)"
            )
            sys.stdout.flush()
            last_log = now

    writer.release()
    elapsed = time.perf_counter() - t_start
    out_size = os.path.getsize(args.output)
    print(f"\n[Prerender] Done in {elapsed:.1f}s ({total/elapsed:.1f} fps)")
    print(f"[Prerender] Output: {out_size:,} bytes "
          f"(ratio {out_size/file_size:.1f}x source)")
    print(f"[Prerender] Total seconds at fps={fps}: {total/fps:.1f}s")
    print(f"\nPlayback: python sender.py --video {args.output} "
          f"--profile {args.profile} --loop --screen 1")


if __name__ == "__main__":
    main()
