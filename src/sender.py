"""Sender - Encodes a file into 4-color QR frame sequence and displays fullscreen"""
import argparse
import sys
import time

import cv2

from config import load_profile
from codec import ColorQRCodec
from display import create_display


def main():
    parser = argparse.ArgumentParser(description="QR File Transfer - Sender")
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to the file to send (optional when --video is specified)")
    parser.add_argument("--video", default=None,
                        help="Play from a video file pre-rendered by prerender.py")
    parser.add_argument("--cellgrid", default=None,
                        help="Play from a file created by prerender.py --cellgrid "
                             "(no RS encode needed, only palette+resize, ultra-fast)")
    parser.add_argument("--profile", default="cvhduvc2",
                        help="Profile name (default: cvhduvc2)")
    parser.add_argument("--display", default="pygame",
                        choices=["pygame", "opencv", "null"],
                        help="Rendering backend (default: pygame)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop playback after all frames are displayed")
    parser.add_argument("--screen", type=int, default=0,
                        help="Display monitor number (default: 0=primary, "
                             "1=extended display for single-PC self-loop)")
    parser.add_argument("--config", default=None,
                        help="Path to profile config file")
    args = parser.parse_args()

    if not args.video and not args.cellgrid and not args.file:
        parser.error("One of --video, --cellgrid, or file path is required")

    config = load_profile(args.profile, args.config)
    display = create_display(config, args.display, screen_index=args.screen)

    print(f"[Sender] Profile: {config.name}")
    print(f"[Sender] Resolution: {config.display.resolution}")
    print(f"[Sender] FPS: {config.display.fps}")

    if args.cellgrid:
        # Play from cell grid intermediate representation (fastest, no RS encode needed)
        run_from_cellgrid(args.cellgrid, config, display, args.loop)
    elif args.video:
        # Play from pre-encoded video (minimal CPU contention, stable display.fps)
        run_from_video(args.video, config, display, args.loop)
    else:
        # Normal mode: on-demand encode from file
        run_from_file(args.file, config, display, args.loop)


def run_from_file(file_path, config, display, do_loop):
    """On-demand encode from file and display"""
    codec = ColorQRCodec(config)
    print(f"[Sender] Cell size: {config.codec.cell_size}px, "
          f"Colors: {config.codec.num_colors}")
    print(f"[Sender] Payload capacity/frame: {codec.payload_capacity} bytes")
    print(f"[Sender] Preparing chunks: {file_path}")
    chunks = codec.prepare_chunks(file_path)
    total = len(chunks)
    print(f"[Sender] Total frames: {total}")

    with display:
        print("[Sender] Displaying frames... (ESC to quit)")
        while True:
            for info in chunks:
                if display.should_quit():
                    break
                frame = codec.encode_chunk(*info)
                display.show_frame(frame)
                sys.stdout.write(f"\r[Sender] Frame {info[0]+1}/{total}")
                sys.stdout.flush()

            if display.should_quit():
                break
            if not do_loop:
                print("\n[Sender] All frames displayed. Waiting 3 seconds...")
                time.sleep(3)
                break
            print("\n[Sender] Looping...")

    print("\n[Sender] Done.")


def run_from_video(video_path, config, display, do_loop):
    """Play from pre-encoded video (no encode needed = minimal CPU contention)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Sender] Video source: {video_path}")
    print(f"[Sender] Source: {src_w}x{src_h}@{src_fps:.1f}fps, {total} frames")
    if (src_w, src_h) != tuple(config.display.resolution):
        print(f"WARN: video resolution {src_w}x{src_h} != "
              f"profile {config.display.resolution}")

    with display:
        print("[Sender] Displaying frames from video... (ESC to quit)")
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
            idx = 0
            while True:
                if display.should_quit():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                display.show_frame(frame)
                idx += 1
                sys.stdout.write(f"\r[Sender] Frame {idx}/{total}")
                sys.stdout.flush()

            if display.should_quit():
                break
            if not do_loop:
                print("\n[Sender] All frames displayed. Waiting 3 seconds...")
                time.sleep(3)
                break
            print("\n[Sender] Looping...")

    cap.release()
    print("\n[Sender] Done.")


def run_from_cellgrid(cellgrid_path, config, display, do_loop):
    """Play from cell grid intermediate file (palette+resize only, fastest)"""
    import struct
    import numpy as np
    codec = ColorQRCodec(config)
    f = open(cellgrid_path, "rb")
    magic = f.read(8)
    if magic != b"QRCELLS\x01":
        print(f"ERROR: bad magic in {cellgrid_path}")
        sys.exit(1)
    gr, gc, total = struct.unpack(">HHI", f.read(8))
    if gr != codec.grid_rows or gc != codec.grid_cols:
        print(f"ERROR: cell grid size mismatch (file {gr}×{gc}, "
              f"profile {codec.grid_rows}×{codec.grid_cols})")
        sys.exit(1)
    frame_size = gr * gc
    body_start = f.tell()
    print(f"[Sender] cellgrid: {gr}×{gc}, {total} frames, "
          f"{frame_size} bytes/frame")

    with display:
        print("[Sender] Displaying frames from cellgrid... (ESC to quit)")
        while True:
            f.seek(body_start)
            for idx in range(total):
                if display.should_quit():
                    break
                grid_bytes = f.read(frame_size)
                if len(grid_bytes) < frame_size:
                    break
                grid = np.frombuffer(grid_bytes, dtype=np.uint8).reshape(gr, gc)
                frame = codec.cell_grid_to_image(grid)
                display.show_frame(frame)
                sys.stdout.write(f"\r[Sender] Frame {idx+1}/{total}")
                sys.stdout.flush()

            if display.should_quit():
                break
            if not do_loop:
                print("\n[Sender] All frames displayed. Waiting 3 seconds...")
                time.sleep(3)
                break
            print("\n[Sender] Looping...")

    f.close()
    print("\n[Sender] Done.")


if __name__ == "__main__":
    main()
