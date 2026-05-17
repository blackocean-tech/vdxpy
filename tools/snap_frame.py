"""Capture a frame, save as PNG, and analyze its structure"""
import argparse
import sys

import numpy as np
import cv2

from config import load_profile
from capture import create_capture


def main():
    parser = argparse.ArgumentParser(description="Snap one frame and analyze")
    parser.add_argument("--profile", default="cvhduvc2")
    parser.add_argument("--device", "-d", type=int, default=2)
    parser.add_argument("--output", "-o", default="captured_frame.png")
    parser.add_argument("--cell-size", type=int, default=None,
                        help="Cell size for analysis (default: from profile)")
    args = parser.parse_args()

    config = load_profile(args.profile)
    config.capture.device_index = args.device
    capture = create_capture(config)
    cell_size = args.cell_size or config.codec.cell_size

    with capture:
        frame = None
        for _ in range(15):
            f = capture.read_frame()
            if f is not None:
                frame = f

    if frame is None:
        print("No frame captured")
        sys.exit(1)

    print(f"Profile: {args.profile} device={args.device}")
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"Mean BGR: {frame.mean(axis=(0,1))}")
    cv2.imwrite(args.output, frame)
    print(f"Saved: {args.output}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    print(f"\nrow means: min={row_mean.min():.1f} max={row_mean.max():.1f} "
          f"std={row_mean.std():.1f}")
    print(f"col means: min={col_mean.min():.1f} max={col_mean.max():.1f} "
          f"std={col_mean.std():.1f}")

    print(f"\n=== {cell_size}x{cell_size} cell block average color (top-left 10x10 cells) ===")
    for r in range(10):
        row_str = []
        for c in range(10):
            y0 = r * cell_size
            x0 = c * cell_size
            block = frame[y0:y0+cell_size, x0:x0+cell_size]
            avg = block.mean(axis=(0,1)).astype(int)
            row_str.append(f"({avg[0]:3d},{avg[1]:3d},{avg[2]:3d})")
        print(" ".join(row_str))

    y_mid = frame.shape[0] // 2
    unique_in_row = set()
    for x in range(0, frame.shape[1], cell_size):
        block = frame[y_mid:y_mid+cell_size, x:x+cell_size]
        avg = tuple((block.mean(axis=(0,1)) > 128).astype(int).tolist())
        unique_in_row.add(avg)
    print(f"\nQuantized color types at row y={y_mid}: {len(unique_in_row)}")
    print(f"  (expected: max 8 for 8-color profile; more variation = signal present)")


if __name__ == "__main__":
    main()
