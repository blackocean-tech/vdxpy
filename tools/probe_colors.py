"""Detect actual colors used in captured images"""
import argparse
import sys
import numpy as np
import cv2

from config import load_profile
from capture import create_capture

parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="cvhduvc2")
parser.add_argument("--device", "-d", type=int, default=2)
args = parser.parse_args()

config = load_profile(args.profile)
config.capture.device_index = args.device
capture = create_capture(config)

with capture:
    frame = None
    for _ in range(10):
        frame = capture.read_frame()
        if frame is not None:
            break
    if frame is None:
        print("No frame")
        sys.exit(1)

cell_size = config.codec.cell_size  # 8
# Collect BGR values at all cell center positions
grid_rows = 1080 // cell_size
grid_cols = 1920 // cell_size
ys = np.arange(grid_rows) * cell_size + cell_size // 2
xs = np.arange(grid_cols) * cell_size + cell_size // 2
yy, xx = np.meshgrid(ys, xs, indexing='ij')
samples = frame[yy.ravel(), xx.ravel()].astype(np.int32)  # (N, 3) BGR

# 8-color clustering (KMeans-like - round to nearest BGR vertex)
# First binarize each BGR component to 0/255 (threshold at 128)
binary = (samples > 128).astype(np.int32) * 255
# Convert to 8-vertex codes
unique, counts = np.unique(binary, axis=0, return_counts=True)
order = np.argsort(-counts)
print(f"Frame shape: {frame.shape}")
print(f"Total cells sampled: {len(samples)}")
print(f"Distinct quantized colors found: {len(unique)}")
print()
print(f"{'BGR (quantized)':<20} {'count':>8} {'%':>6}  {'mean BGR (raw)':<25}")
total = len(samples)
for i in order:
    bgr = unique[i].tolist()
    c = counts[i]
    # Mean of raw samples classified as this color
    mask = np.all(binary == unique[i], axis=1)
    mean_raw = samples[mask].mean(axis=0).astype(int).tolist()
    pct = c / total * 100
    name = ""
    bgr_t = tuple(bgr)
    color_names = {
        (0,0,0): "Black",
        (0,0,255): "Red(B0G0R255)",
        (0,255,0): "Green",
        (255,0,0): "Blue(B255G0R0)",
        (0,255,255): "Yellow(BGR=0,255,255)",
        (255,0,255): "Magenta",
        (255,255,0): "Cyan(BGR=255,255,0)",
        (255,255,255): "White",
    }
    name = color_names.get(bgr_t, "?")
    print(f"{str(bgr):<20} {c:>8} {pct:>5.1f}%  {str(mean_raw):<25} {name}")
