"""Base class definitions: abstract interfaces for capture, codec, and display"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import ProfileConfig


@dataclass
class FrameData:
    """Encoded frame data"""
    frame_number: int
    payload: bytes
    crc: int
    total_frames: int
    is_last: bool = False


class BaseCapture(ABC):
    """Abstract interface for the capture layer"""

    def __init__(self, config: ProfileConfig):
        self.config = config

    @abstractmethod
    def open(self) -> bool:
        """Open the device. Returns True on success."""
        ...

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame. Returns a BGR ndarray, or None on failure."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release the device"""
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()


class BaseCodec(ABC):
    """Abstract interface for the codec layer"""

    def __init__(self, config: ProfileConfig):
        self.config = config

    @abstractmethod
    def encode_file(self, file_path: str) -> List[np.ndarray]:
        """Encode a file into a sequence of frames (list of BGR images)"""
        ...

    @abstractmethod
    def decode_frame(self, image: np.ndarray) -> Optional[FrameData]:
        """Decode a single frame image into FrameData. Returns None on failure."""
        ...

    @abstractmethod
    def reassemble(self, frames: dict) -> bytes:
        """Reassemble file data from a frame dict {frame_number: FrameData}"""
        ...

    def get_grid_size(self) -> tuple:
        """Return the grid size (cols, rows) of the display area"""
        w, h = self.config.display.resolution
        cell = self.config.codec.cell_size
        cols = w // cell
        rows = h // cell
        return cols, rows

    def get_data_capacity_per_frame(self) -> int:
        """Return the data capacity per frame in bytes"""
        cols, rows = self.get_grid_size()
        total_cells = cols * rows

        if self.config.features.use_anchors:
            anchor_cells = 3 * 8 * 8
            total_cells -= anchor_cells
        if self.config.features.reference_patches:
            total_cells -= 4 * 4

        bits_per_cell = {2: 1, 4: 2, 8: 3}.get(self.config.codec.num_colors, 2)
        total_bits = total_cells * bits_per_cell
        total_bytes = total_bits // 8

        header_size = self.config.codec.frame_header_size
        ecc_overhead = int(total_bytes * self.config.codec.ecc_redundancy)

        return max(1, total_bytes - header_size - ecc_overhead)


class BaseDisplay(ABC):
    """Abstract interface for the display layer"""

    def __init__(self, config: ProfileConfig):
        self.config = config

    @abstractmethod
    def open(self) -> bool:
        """Initialize the display. Returns True on success."""
        ...

    @abstractmethod
    def show_frame(self, image: np.ndarray) -> None:
        """Display a frame image (BGR ndarray) on screen"""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the display"""
        ...

    @abstractmethod
    def should_quit(self) -> bool:
        """Check if the user has requested to quit"""
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
