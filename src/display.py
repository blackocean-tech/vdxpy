"""Display layer - frame rendering via pygame/OpenCV"""
from typing import Optional

import cv2
import numpy as np

from base import BaseDisplay
from config import ProfileConfig


class PygameDisplay(BaseDisplay):
    """Fullscreen/windowed display using pygame"""

    def __init__(self, config: ProfileConfig, screen_index: int = 0):
        super().__init__(config)
        self.screen = None
        self.clock = None
        self._quit_requested = False
        self._screen_index = screen_index  # Target display in multi-monitor setups

    def open(self) -> bool:
        import pygame
        pygame.init()
        w, h = self.config.display.resolution

        flags = 0
        if self.config.display.fullscreen:
            flags |= pygame.FULLSCREEN
        if self.config.display.vsync:
            flags |= pygame.DOUBLEBUF | pygame.HWSURFACE

        # Single-PC loop: select display (e.g. display=1 for secondary)
        n_displays = pygame.display.get_num_displays()
        if 0 <= self._screen_index < n_displays:
            self.screen = pygame.display.set_mode(
                (w, h), flags, display=self._screen_index
            )
        else:
            self.screen = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("QR File Transfer - Sender")
        self.clock = pygame.time.Clock()
        self._quit_requested = False
        return True

    def show_frame(self, image: np.ndarray) -> None:
        import pygame
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.config.display.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit_requested = True

    def close(self) -> None:
        import pygame
        pygame.quit()

    def should_quit(self) -> bool:
        return self._quit_requested


class OpenCVDisplay(BaseDisplay):
    """Simple display using OpenCV imshow (fallback when pygame is unavailable)"""

    def __init__(self, config: ProfileConfig):
        super().__init__(config)
        self._quit_requested = False
        self.window_name = "QR File Transfer"

    def open(self) -> bool:
        if self.config.display.fullscreen:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self._quit_requested = False
        return True

    def show_frame(self, image: np.ndarray) -> None:
        cv2.imshow(self.window_name, image)
        wait_ms = max(1, int(1000 / self.config.display.fps))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27 or key == ord('q'):
            self._quit_requested = True

    def close(self) -> None:
        cv2.destroyAllWindows()

    def should_quit(self) -> bool:
        return self._quit_requested


class NullDisplay(BaseDisplay):
    """No display (for testing and benchmarking)"""

    def __init__(self, config: ProfileConfig):
        super().__init__(config)

    def open(self) -> bool:
        return True

    def show_frame(self, image: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass

    def should_quit(self) -> bool:
        return False


def create_display(config: ProfileConfig, backend: str = "pygame",
                   screen_index: int = 0) -> BaseDisplay:
    if backend == "pygame":
        return PygameDisplay(config, screen_index=screen_index)
    elif backend == "opencv":
        return OpenCVDisplay(config)
    elif backend == "null":
        return NullDisplay(config)
    else:
        return PygameDisplay(config, screen_index=screen_index)
