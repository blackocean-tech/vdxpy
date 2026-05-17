"""Capture quality diagnostic tool - measures cell_size/color count/ECC margin"""
import argparse
import sys
import time

import numpy as np
try:
    from creedsolo import ReedSolomonError
except ImportError:
    from reedsolo import ReedSolomonError

from config import load_profile
from codec import ColorQRCodec, HEADER_SIZE
from capture import create_capture


def diagnose_frame(codec: ColorQRCodec, image: np.ndarray):
    """Decode one frame and return quality metrics.
    Returns: (success, margins, rs_errors_per_block, cell_max_dist) or None
    """
    h_img, w_img = image.shape[:2]
    ys = np.clip(codec._cell_y, 0, h_img - 1)
    xs = np.clip(codec._cell_x, 0, w_img - 1)
    sampled = image[ys, xs].astype(np.int32)
    diff = sampled[:, None, :] - codec._ref_bgr[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    sorted_dist_sq = np.sort(dist_sq, axis=1)
    # Margin: distance difference between 2nd-nearest and nearest color (sqrt)
    nearest = np.sqrt(sorted_dist_sq[:, 0])
    second = np.sqrt(sorted_dist_sq[:, 1])
    margins = second - nearest  # larger = safer (positive: correct, negative: impossible)
    cell_values_arr = np.argmin(dist_sq, axis=1).astype(np.int32)
    cell_values = cell_values_arr.tolist()

    raw_byte_count = (len(codec.data_cell_positions) * codec.bits_per_cell) // 8
    raw_bytes = codec._cells_to_bytes(cell_values, raw_byte_count)

    # Perform RS decode per-block and get error count
    rs_errors = []
    decoded_blocks = []
    for i in range(codec.num_blocks):
        start = i * codec.rs_block_size
        block = raw_bytes[start:start + codec.rs_block_size]
        if len(block) < codec.rs_block_size:
            block = block.ljust(codec.rs_block_size, b'\x00')
        try:
            result = codec.rs.decode(block)
            decoded = bytes(result[0]) if isinstance(result, tuple) else bytes(result)
            # if errata_pos exists, its length is the error count
            errata = result[2] if isinstance(result, tuple) and len(result) > 2 else []
            rs_errors.append(len(errata))
            decoded_blocks.append(decoded)
        except ReedSolomonError:
            rs_errors.append(-1)  # uncorrectable
            decoded_blocks.append(None)

    if any(d is None for d in decoded_blocks):
        return {"success": False, "margins": margins, "rs_errors": rs_errors,
                "nearest": nearest, "second": second}

    decoded = b"".join(decoded_blocks)
    if len(decoded) < HEADER_SIZE:
        return {"success": False, "margins": margins, "rs_errors": rs_errors,
                "nearest": nearest, "second": second}

    import struct
    frame_num, total_frames, payload_len, flags, _ = struct.unpack(">IHHBB",
                                                                    decoded[:HEADER_SIZE])
    valid_header = (total_frames > 0 and total_frames <= 0xFFFF
                    and frame_num < total_frames)
    return {"success": valid_header, "margins": margins, "rs_errors": rs_errors,
            "nearest": nearest, "second": second, "frame_num": frame_num,
            "total_frames": total_frames}


def main():
    parser = argparse.ArgumentParser(description="QR Capture Quality Diagnosis")
    parser.add_argument("--device", "-d", type=int, default=2)
    parser.add_argument("--profile", default="cvhduvc2")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_profile(args.profile, args.config)
    config.capture.device_index = args.device
    codec = ColorQRCodec(config)

    print(f"[Diagnose] Profile: {config.name}")
    print(f"[Diagnose] cell_size={codec.cell_size}, num_colors={codec.num_colors}, "
          f"ecc_redundancy={codec.ecc_redundancy}")
    print(f"[Diagnose] num_blocks={codec.num_blocks}, rs_block_size={codec.rs_block_size}, "
          f"nsym={codec.nsym}")
    print(f"[Diagnose] data_cells={len(codec.data_cell_positions)}, "
          f"payload/frame={codec.payload_capacity} bytes")
    print(f"[Diagnose] Reference colors (BGR int): {codec._ref_bgr.tolist()}")

    capture = create_capture(config)
    aggregated_margins = []
    aggregated_nearest = []
    aggregated_rs_errors = []
    success_count = 0
    fail_count = 0

    with capture:
        print(f"[Diagnose] Capturing {args.num_frames} successful frames...")
        attempts = 0
        max_attempts = args.num_frames * 5
        while success_count < args.num_frames and attempts < max_attempts:
            attempts += 1
            frame = capture.read_frame()
            if frame is None:
                continue
            result = diagnose_frame(codec, frame)
            if result is None:
                continue
            if result["success"]:
                success_count += 1
                aggregated_margins.append(result["margins"])
                aggregated_nearest.append(result["nearest"])
                aggregated_rs_errors.extend(result["rs_errors"])
                sys.stdout.write(f"\r[Diagnose] OK frames: {success_count}/{args.num_frames}")
                sys.stdout.flush()
            else:
                fail_count += 1

    print()
    print(f"[Diagnose] success={success_count}, fail={fail_count}, attempts={attempts}")
    if not aggregated_margins:
        print("[Diagnose] No successful frames captured.")
        return

    margins_all = np.concatenate(aggregated_margins)
    nearest_all = np.concatenate(aggregated_nearest)
    rs_errors_arr = np.array(aggregated_rs_errors)

    print()
    print("=== Cell Color Margin (2nd-nearest minus nearest, in BGR Euclidean dist) ===")
    print(f"  min={margins_all.min():.1f}  mean={margins_all.mean():.1f}  "
          f"median={np.median(margins_all):.1f}  max={margins_all.max():.1f}")
    print(f"  percentile 1%={np.percentile(margins_all, 1):.1f}, "
          f"5%={np.percentile(margins_all, 5):.1f}, 10%={np.percentile(margins_all, 10):.1f}")
    # Ratio of danger cells (small margin)
    danger_threshold = 30  # BGR Euclidean distance < 30 is somewhat risky
    danger_ratio = (margins_all < danger_threshold).mean() * 100
    print(f"  cells with margin<{danger_threshold}: {danger_ratio:.2f}%")

    print()
    print("=== Cell distance to nearest reference color ===")
    print(f"  min={nearest_all.min():.1f}  mean={nearest_all.mean():.1f}  "
          f"max={nearest_all.max():.1f}  p99={np.percentile(nearest_all, 99):.1f}")

    print()
    print("=== Reed-Solomon errors per block (positive=corrected, -1=uncorrectable) ===")
    print(f"  blocks={len(rs_errors_arr)}, max_correctable={codec.nsym // 2}")
    print(f"  min={rs_errors_arr.min()}  mean={rs_errors_arr.mean():.2f}  "
          f"max={rs_errors_arr.max()}  p99={np.percentile(rs_errors_arr, 99):.1f}")
    print(f"  uncorrectable blocks: {(rs_errors_arr == -1).sum()}")
    used_ratio = (rs_errors_arr.clip(0) / (codec.nsym // 2)).mean() * 100
    print(f"  ECC capacity used (avg): {used_ratio:.2f}%")
    p99_used = np.percentile(rs_errors_arr.clip(0), 99) / (codec.nsym // 2) * 100
    print(f"  ECC capacity used (p99): {p99_used:.2f}%")


if __name__ == "__main__":
    main()
