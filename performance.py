"""Thread-safe per-capture performance measurements and reports."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Generator


@dataclass(frozen=True)
class StageTiming:
    name: str
    started_at: datetime
    ended_at: datetime
    seconds: float


class PerformanceSession:
    """Collect stage timings that may be written by different worker threads."""

    REPORT_ORDER = (
        "Camera Capture", "Quality Check", "Document Detection",
        "Perspective Transform", "Upscale", "Preprocess", "EasyOCR",
        "OCR Merge", "Gemini API", "AI Postprocess", "TTS Generate",
        "Audio Play",
    )

    def __init__(self, job_id: int, logger: logging.Logger) -> None:
        self.job_id, self.logger = job_id, logger
        # Keep performance records enabled even when general DEBUG_MODE is off.
        self.perf_logger = logging.getLogger(f"{logger.name}.performance")
        self.perf_logger.setLevel(logging.INFO)
        self.started_at = datetime.now()
        self._started_perf = time.perf_counter()
        self._timings: list[StageTiming] = []
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        started_at, started = datetime.now(), time.perf_counter()
        self.perf_logger.info("PERF job=%s stage=%s start=%s", self.job_id, name,
                              started_at.isoformat(timespec="milliseconds"))
        try:
            yield
        finally:
            ended_at = datetime.now()
            seconds = time.perf_counter() - started
            with self._lock:
                self._timings.append(StageTiming(name, started_at, ended_at, seconds))
            self.perf_logger.info("PERF job=%s stage=%s end=%s duration=%.3f sec",
                                  self.job_id, name, ended_at.isoformat(timespec="milliseconds"), seconds)

    def report(self) -> str:
        total = time.perf_counter() - self._started_perf
        with self._lock:
            timings = list(self._timings)
        totals = {name: sum(item.seconds for item in timings if item.name == name)
                  for name in self.REPORT_ORDER}
        measured = [(name, seconds) for name, seconds in totals.items() if seconds > 0]
        longest_name, longest_seconds = max(measured, key=lambda item: item[1], default=("N/A", 0.0))
        percent = longest_seconds / total * 100 if total else 0.0
        lines = ["=" * 28, "Performance Report", "=" * 28]
        for name in self.REPORT_ORDER:
            lines.extend([name, f"{totals[name]:.2f} sec"])
        lines.extend(["Total Processing Time", f"{total:.2f} sec", "=" * 28,
                      "Longest Stage", longest_name, f"{percent:.1f}%", "=" * 28])
        return "\n".join(lines)
