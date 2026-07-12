"""OpenCV camera acquisition and preview UI."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import Config


class Camera:
    """Own a camera device and provide an accessible capture preview."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        """Open and configure the selected camera."""
        self.capture = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.logger.error("Camera %s could not be opened", self.config.camera_index)
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return True

    def wait_for_capture(self, status: str = "SPACE : Capture   ESC : Exit") -> np.ndarray | None:
        """Display live preview until SPACE returns a frame or ESC returns None."""
        if self.capture is None:
            return None
        while True:
            ok, frame = self.capture.read()
            if not ok:
                self.logger.error("Could not read a camera frame")
                return None
            preview = frame.copy()
            cv2.putText(preview, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("Reading Assistant", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return None
            if key == ord(" "):
                return frame

    def save_capture(self, frame: np.ndarray) -> Path | None:
        """Persist a captured frame and return its path."""
        path = self.config.capture_dir / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        try:
            if cv2.imwrite(str(path), frame):
                return path
        except cv2.error:
            self.logger.exception("Capture save failed")
        self.logger.error("Capture save failed: %s", path)
        return None

    def close(self) -> None:
        """Release the camera and close OpenCV windows."""
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
