"""Profile configuration loader"""
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CaptureConfig:
    backend: str = "mjpeg"
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    device_index: int = 0
    # Optimization #2: raw YUY2 read (MSMF + CONVERT_RGB=0)
    # When True, skips BGR conversion and returns raw YUY2 with shape (H, W, 2).
    # The codec decodes via direct Y-channel LUT when using a grayscale palette.
    raw_yuy2: bool = False


@dataclass
class CodecConfig:
    cell_size: int = 16
    num_colors: int = 4
    colors: List[List[int]] = field(default_factory=lambda: [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]
    ])
    ecc_redundancy: float = 0.25
    frame_header_size: int = 8
    # Frame-level erasure code (M1): K data + M parity per group
    fec_M_per_group: int = 0  # 0=disabled, >0 adds M parity frames per group
    fec_group_size: int = 200  # group_size + M <= 255 required (GF(2^8) constraint)
    # Header v2: payload_len as u32 (needed when cell_size <= 2)
    header_v2: bool = False


@dataclass
class DisplayConfig:
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    vsync: bool = True
    fullscreen: bool = True


@dataclass
class FeaturesConfig:
    use_anchors: bool = False
    perspective_correction: bool = False
    color_normalization: bool = False
    reference_patches: bool = False


@dataclass
class ProfileConfig:
    name: str = ""
    description: str = ""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    throughput_estimate: str = ""


def load_profile(profile_name: str, config_path: str = None) -> ProfileConfig:
    if config_path is None:
        # Look for individual profile YAML in profiles/ directory first
        profiles_dir = os.path.join(os.path.dirname(__file__), "..", "profiles")
        per_file = os.path.join(profiles_dir, f"{profile_name}.yaml")
        if os.path.exists(per_file):
            config_path = per_file
        else:
            # Fallback: single profiles.yaml in src/ or project root
            config_path = os.path.join(os.path.dirname(__file__), "profiles.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(
                    os.path.dirname(__file__), "..", "profiles.yaml"
                )

    with open(config_path, "r", encoding="utf-8") as f:
        all_profiles = yaml.safe_load(f)

    if profile_name not in all_profiles:
        available = ", ".join(all_profiles.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")

    p = all_profiles[profile_name]

    cap = p.get("capture", {})
    cod = p.get("codec", {})
    disp = p.get("display", {})
    feat = p.get("features", {})

    return ProfileConfig(
        name=profile_name,
        description=p.get("description", ""),
        capture=CaptureConfig(
            backend=cap.get("backend", "mjpeg"),
            resolution=tuple(cap.get("resolution", [1920, 1080])),
            fps=cap.get("fps", 30),
            device_index=cap.get("device_index", 0),
            raw_yuy2=cap.get("raw_yuy2", False),
        ),
        codec=CodecConfig(
            cell_size=cod.get("cell_size", 16),
            num_colors=cod.get("num_colors", 4),
            colors=cod.get("colors", [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]),
            ecc_redundancy=cod.get("ecc_redundancy", 0.25),
            frame_header_size=cod.get("frame_header_size", 8),
            fec_M_per_group=cod.get("fec_M_per_group", 0),
            fec_group_size=cod.get("fec_group_size", 200),
            header_v2=cod.get("header_v2", False),
        ),
        display=DisplayConfig(
            resolution=tuple(disp.get("resolution", [1920, 1080])),
            fps=disp.get("fps", 30),
            vsync=disp.get("vsync", True),
            fullscreen=disp.get("fullscreen", True),
        ),
        features=FeaturesConfig(
            use_anchors=feat.get("use_anchors", False),
            perspective_correction=feat.get("perspective_correction", False),
            color_normalization=feat.get("color_normalization", False),
            reference_patches=feat.get("reference_patches", False),
        ),
        throughput_estimate=p.get("throughput_estimate", ""),
    )
