"""Capture layer - OpenCV VideoCapture wrapper + mock/recorded playback"""
import os
import glob
import queue
import threading
from typing import Optional

import cv2
import numpy as np

from base import BaseCapture
from config import ProfileConfig


class OpenCVCapture(BaseCapture):
    """Real device capture using OpenCV VideoCapture"""

    def __init__(self, config: ProfileConfig):
        super().__init__(config)
        self.cap = None
        self._raw_yuy2 = False

    def open(self) -> bool:
        idx = self.config.capture.device_index
        backend = self.config.capture.backend
        raw_yuy2_req = (backend == "yuy2"
                        and getattr(self.config.capture, "raw_yuy2", False))

        # Optimization #2: YUY2 raw mode uses MSMF + CONVERT_RGB=0 to get raw YUY2.
        # DSHOW ignores CONVERT_RGB=0 so it cannot be used.
        if raw_yuy2_req:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
            self._raw_yuy2 = True
        elif backend == "yuy2":
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        else:
            self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        w, h = self.config.capture.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.capture.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mode = " (raw YUY2)" if self._raw_yuy2 else ""
        print(f"[OpenCVCapture]{mode} requested {w}x{h}@{self.config.capture.fps}fps, "
              f"got {actual_w}x{actual_h}@{actual_fps:.1f}fps")

        self._needs_resize = (actual_w != w and actual_w != w * 2) or (actual_h != h)
        self._target_size = (w, h)

        return self.cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self._raw_yuy2:
            # Raw YUY2 is returned as (1, W*H*2), reshape to (H, W, 2)
            w, h = self.config.capture.resolution
            frame = frame.reshape(h, w, 2)
            return frame
        if self._needs_resize:
            frame = cv2.resize(frame, self._target_size)
        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def drain(self, max_grabs: int = 5) -> int:
        """Countermeasure 2: Call cap.grab() multiple times to discard stale frames in buffer.

        Used when OpenCV's BUFFERSIZE=1 is ineffective for the backend, or when
        the buffer has grown due to slow decoding. Lightweight since retrieve() is not called.
        """
        if self.cap is None:
            return 0
        n = 0
        for _ in range(max_grabs):
            if not self.cap.grab():
                break
            n += 1
        return n


class ThreadedCapture(BaseCapture):
    """Countermeasure 1: Continuously capture in background thread, providing only the latest frame.

    Capture is never blocked regardless of decode processing time,
    preventing OpenCV internal buffer accumulation. read_frame() always returns the
    latest frame, and stale frames read too late are silently discarded.
    """

    def __init__(self, config: ProfileConfig):
        super().__init__(config)
        self._cap = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._captured_total = 0  # Total captured count (including discarded)
        self._dropped = 0         # Count of frames consumer missed
        self._raw_yuy2 = False

    def open(self) -> bool:
        idx = self.config.capture.device_index
        backend = self.config.capture.backend
        raw_yuy2_req = (backend == "yuy2"
                        and getattr(self.config.capture, "raw_yuy2", False))
        if raw_yuy2_req:
            # Optimization #2: YUY2 raw mode (MSMF + CONVERT_RGB=0)
            self._cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
            self._raw_yuy2 = True
        elif backend == "yuy2":
            self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        else:
            self._cap = cv2.VideoCapture(idx)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        w, h = self.config.capture.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.capture.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mode = " (raw YUY2)" if self._raw_yuy2 else ""
        print(f"[ThreadedCapture]{mode} requested {w}x{h}@{self.config.capture.fps}fps, "
              f"got {actual_w}x{actual_h}@{actual_fps:.1f}fps")
        if not self._cap.isOpened():
            return False
        self._needs_resize = (actual_w != w and actual_w != w * 2) or (actual_h != h)
        self._target_size = (w, h)
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        w, h = self.config.capture.resolution
        while not self._stop_evt.is_set():
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._raw_yuy2:
                # Raw YUY2 is returned as (1, W*H*2), reshape to (H, W, 2)
                frame = frame.reshape(h, w, 2)
            elif self._needs_resize:
                frame = cv2.resize(frame, self._target_size)
            self._captured_total += 1
            # Discard old frame and put the latest one
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def read_frame(self) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=2.0)
        except queue.Empty:
            return None

    def release(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def stats(self) -> dict:
        return {"captured_total": self._captured_total, "dropped": self._dropped}


class MockCapture(BaseCapture):
    """Mock capture - reads from frame sequence image files or in-memory buffer"""

    def __init__(self, config: ProfileConfig, frames: list = None,
                 frame_dir: str = None):
        super().__init__(config)
        self._frames = frames or []
        self._frame_dir = frame_dir
        self._index = 0
        self._loaded = False

    def open(self) -> bool:
        if self._frame_dir and not self._frames:
            pattern = os.path.join(self._frame_dir, "frame_*.png")
            paths = sorted(glob.glob(pattern))
            self._frames = [cv2.imread(p) for p in paths if cv2.imread(p) is not None]
        self._index = 0
        self._loaded = True
        return len(self._frames) > 0

    def set_frames(self, frames: list):
        self._frames = frames
        self._index = 0

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._loaded or self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def release(self) -> None:
        self._frames = []
        self._index = 0
        self._loaded = False


class FileCapture(BaseCapture):
    """Capture from video file (for recorded playback)"""

    def __init__(self, config: ProfileConfig, video_path: str):
        super().__init__(config)
        self.video_path = video_path
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        w, h = self.config.capture.resolution
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


def create_capture(config: ProfileConfig, **kwargs) -> BaseCapture:
    backend = config.capture.backend
    if backend == "mock":
        return MockCapture(config, **kwargs)
    elif "video_path" in kwargs:
        return FileCapture(config, kwargs["video_path"])
    elif kwargs.get("threaded"):
        return ThreadedCapture(config)
    else:
        return OpenCVCapture(config)
