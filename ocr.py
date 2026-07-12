"""EasyOCR abstraction and OCR quality evaluation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import Config


@dataclass(frozen=True)
class OCRItem:
    """One EasyOCR recognition result."""
    box: list[list[int]] | None  # Modified: paragraph results may not include a box.
    text: str
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    """Combined OCR text, individual regions, and derived quality metrics."""
    items: list[OCRItem]
    text: str
    average_confidence: float


class OCRService:
    """Lazy EasyOCR reader, deliberately isolated for later engine replacement."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger
        self._reader: object | None = None
        self._debug_result_logged = False  # Modified: log EasyOCR output shape once.

    def initialize(self) -> bool:
        """Load EasyOCR models, returning False instead of terminating on error."""
        try:
            import easyocr
            self._reader = easyocr.Reader(list(self.config.ocr_languages), gpu=self.config.ocr_gpu)
            return True
        except Exception:
            self.logger.exception("EasyOCR initialization failed")
            return False

    def has_text(self, image: np.ndarray) -> bool:
        """Use connected components as a fast pre-OCR text-presence heuristic."""
        import cv2
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else []
        regions = sum(12 <= int(area) <= 5000 for area in areas)
        self.logger.info("Text presence regions: %s", regions)
        return regions >= self.config.min_text_regions

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize paragraph text and retain box/confidence values."""
        if self._reader is None:
            raise RuntimeError("OCR service is not initialized")
        raw = self._reader.readtext(image, detail=1, paragraph=True)  # type: ignore[union-attr]
        # Modified: EasyOCR result shapes differ by version and paragraph/detail settings.
        if not self._debug_result_logged:
            import easyocr
            # Modified: direct single tuples are logged as one result, not as their box/text field.
            first_item = raw if self._is_result_tuple(raw) else (raw[0] if isinstance(raw, (list, tuple)) and raw else None)
            self.logger.info("EasyOCR: %s", getattr(easyocr, "__version__", "unknown"))
            self.logger.info("Return type: %s", type(raw).__name__)
            self.logger.info("First item: %r", first_item)
            self._debug_result_logged = True

        # Modified: avoid fixed tuple unpacking; empty/None results become an empty list.
        raw_items: list[Any]
        if raw is None:
            raw_items = []
        elif isinstance(raw, str) or self._is_result_tuple(raw):
            raw_items = [raw]
        else:
            try:
                raw_items = list(raw)
            except TypeError:
                raw_items = [raw]

        items: list[OCRItem] = []
        for item in raw_items:  # Modified: normalize every supported EasyOCR shape.
            normalized = self._normalize_item(item)
            if normalized is not None and normalized.text:
                items.append(normalized)
        text = "\n".join(item.text for item in items)
        average = sum(item.confidence for item in items) / len(items) if items else 0.0
        self.logger.info("OCR: chars=%s, average_confidence=%.3f", len(text), average)
        return OCRResult(items=items, text=text, average_confidence=average)

    @staticmethod
    def _is_box(value: Any) -> bool:
        """Return whether a value looks like an EasyOCR quadrilateral box."""
        # Modified: distinguish (box, text) from (text, confidence).
        return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, str)

    @classmethod
    def _is_result_tuple(cls, value: Any) -> bool:
        """Return whether one tuple/list is itself an OCR result."""
        # Modified: support a reader returning one result instead of a list of results.
        return (isinstance(value, tuple) and 1 <= len(value) <= 3
                and (isinstance(value[0], str) or cls._is_box(value[0])))

    @staticmethod
    def _normalize_box(value: Any) -> list[list[int]] | None:
        """Convert a box to integer points, or return None for missing/invalid boxes."""
        # Modified: box data is optional in some EasyOCR output formats.
        if not OCRService._is_box(value):
            return None
        try:
            return [[int(coordinate) for coordinate in point] for point in value]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_item(cls, item: Any) -> OCRItem | None:
        """Normalize supported EasyOCR result variants into one OCRItem."""
        # Modified: accepts (box, text, confidence), (box, text),
        # (text, confidence), and text without fixed unpacking.
        box: list[list[int]] | None = None
        text: Any = ""
        confidence = 1.0
        if isinstance(item, str):
            text = item
        elif isinstance(item, (tuple, list)):
            if len(item) == 3 and cls._is_box(item[0]):
                box, text, confidence = cls._normalize_box(item[0]), item[1], item[2]
            elif len(item) == 2 and cls._is_box(item[0]):
                box, text = cls._normalize_box(item[0]), item[1]
            elif len(item) == 2 and isinstance(item[0], str):
                text, confidence = item[0], item[1]
            elif len(item) == 1:
                text = item[0]
            else:
                return None
        else:
            return None
        if not isinstance(text, str):
            return None
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        return OCRItem(box=box, text=text.strip(), confidence=confidence)

    def is_acceptable(self, result: OCRResult) -> bool:
        """Evaluate OCR using both confidence and useful-character count."""
        return (result.average_confidence >= self.config.min_ocr_confidence
                and len(result.text.replace(" ", "").replace("\n", "")) >= self.config.min_ocr_characters)
