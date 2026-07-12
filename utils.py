"""Shared filesystem, logging, timing, and image helpers."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from config import Config


def ensure_directories(config: Config) -> None:
    """Create runtime output directories if they do not exist."""
    for directory in (config.capture_dir, config.debug_dir, config.log_dir):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(config: Config) -> logging.Logger:
    """Configure a timestamped log file and console logger for this execution."""
    logger = logging.getLogger("reading_assistant")
    if logger.handlers:
        return logger
    # Modified: DEBUG_MODE shows diagnostics; normal mode retains only actionable errors.
    logger.setLevel(logging.DEBUG if config.debug_mode else logging.ERROR)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    log_path = config.log_dir / f"reading_assistant_{datetime.now():%Y%m%d_%H%M%S}.log"  # Modified
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if config.debug_mode:  # Modified
        logger.debug("Debug mode enabled. Log file: %s", log_path)
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str) -> Generator[None, None, None]:
    """Log execution time for a processing step, including failures."""
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s completed in %.3f seconds", label, time.perf_counter() - started)  # Modified: emitted only in debug mode.


def save_debug_image(path: Path, image: np.ndarray, logger: logging.Logger) -> None:
    """Save an image while containing disk/codec failures."""
    try:
        if not cv2.imwrite(str(path), image):
            logger.error("Could not save image: %s", path)
    except cv2.error:
        logger.exception("Image save error: %s", path)
