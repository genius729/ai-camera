"""EasyOCR recognition, multi-image evaluation, and confidence-based merging."""
from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from config import Config
from performance import PerformanceSession
from text_reconstructor import TextReconstructor


@dataclass(frozen=True)
class OCRItem:
    """One normalized EasyOCR recognition region."""
    box: list[list[int]] | None
    text: str
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    """OCR text, quality data, and details of the selected preprocessing variant."""
    items: list[OCRItem]
    text: str
    average_confidence: float
    method: str = ""
    variant_scores: dict[str, tuple[float, int, int]] = field(default_factory=dict)
    low_confidence_words: list[str] = field(default_factory=list)
    variants: dict[str, list[OCRItem]] = field(default_factory=dict)


class OCRService:
    """Lazy EasyOCR service with robust return normalization and ensemble selection."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger
        self._reader: Any | None = None
        self._debug_result_logged = False
        self._reconstructor = TextReconstructor()

    def initialize(self) -> bool:
        """Load EasyOCR in configured CPU/GPU mode without terminating the app."""
        try:
            import easyocr
            self._reader = easyocr.Reader(list(self.config.ocr_languages), gpu=self.config.ocr_gpu)
            return True
        except Exception:
            self.logger.exception("EasyOCR initialization failed")
            return False

    def has_text(self, image: np.ndarray) -> bool:
        """Use connected components as a fast pre-OCR text-presence heuristic."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else []
        regions = sum(12 <= int(area) <= 5000 for area in areas)
        self.logger.info("Text presence regions: %s", regions)
        return regions >= self.config.min_text_regions

    def recognize(self, image: np.ndarray, method: str = "") -> OCRResult:
        """Recognize one image while retaining text, boxes, and confidences."""
        if self._reader is None:
            raise RuntimeError("OCR service is not initialized")
        # paragraph=False retains EasyOCR's real per-region confidence.  With
        # paragraph=True EasyOCR omits it and the old normalizer assigned 1.0.
        raw = self._reader.readtext(image, detail=1, paragraph=False,
                                    canvas_size=self.config.easyocr_canvas_size)
        self._log_first_response(raw)
        normalized = [result for result in (self._normalize_item(item) for item in self._as_items(raw)) if result and result.text]
        raw_text = "\n".join(item.text for item in normalized)
        text = self._reconstructor.reconstruct(normalized) or raw_text
        if self.config.debug_mode:
            self.logger.info("===== OCR RAW =====\n%s\n===== RECONSTRUCTED =====\n%s",
                             raw_text, text)
        average = sum(item.confidence for item in normalized) / len(normalized) if normalized else 0.0
        return OCRResult(normalized, text, average, method, low_confidence_words=self._low_words(normalized))

    def recognize_variants(self, variants: dict[str, np.ndarray],
                           performance: PerformanceSession | None = None) -> OCRResult:
        """OCR all variants, select the best score, and merge highest-confidence words."""
        results = []
        for method, image in variants.items():
            if performance:
                with performance.measure("EasyOCR"):
                    results.append(self.recognize(image, method))
            else:
                results.append(self.recognize(image, method))
            # A trustworthy Original result needs no rescue ensemble.  When it
            # is weak, retain the complete eight-variant accuracy fallback.
            if len(results) == 1 and self.is_acceptable(results[0]):
                first = results[0]
                scores = {first.method: (first.average_confidence,
                                         self._character_count(first.text), len(first.text))}
                self.logger.info("OCR fast path: Original passed confidence/length thresholds")
                return OCRResult(first.items, first.text, first.average_confidence, first.method,
                                 scores, first.low_confidence_words,
                                 {first.method: first.items})
        if not results:
            return OCRResult([], "", 0.0)
        scores = {result.method: (result.average_confidence, self._character_count(result.text), len(result.text))
                  for result in results}
        for method, score in scores.items():
            self.logger.info("OCR variant [%s]: confidence=%.3f, characters=%s, length=%s", method, *score)
            variant = next(result for result in results if result.method == method)
            self.logger.info("OCR variant text [%s]: %s", method, variant.text)
        if self.config.debug_mode:
            self._save_variant_results(results)
        if performance:
            with performance.measure("OCR Merge"):
                selected = max(results, key=lambda value: scores[value.method])
                merged_text, merged_confidence, low_words = self._merge_words(results, selected)
        else:
            selected = max(results, key=lambda value: scores[value.method])
            merged_text, merged_confidence, low_words = self._merge_words(results, selected)
        self.logger.info("Selected OCR variant: %s (confidence=%.3f)", selected.method, selected.average_confidence)
        all_variants = {result.method: result.items for result in results}
        # The selected variant supplies consistent geometry. Confidence/low-word
        # evidence still benefits from the ensemble, while text remains spatially reconstructed.
        return OCRResult(selected.items, selected.text, merged_confidence, selected.method, scores, low_words,
                         all_variants)

    def _save_variant_results(self, results: list[OCRResult]) -> None:
        """Persist every OCR region as JSON during debug sessions."""
        payload = [{
            "preprocessing": result.method,
            "text": result.text,
            "average_confidence": result.average_confidence,
            "regions": [{"text": item.text, "confidence": item.confidence, "box": item.box}
                        for item in result.items],
        } for result in results]
        try:
            path = self.config.debug_dir / "ocr_results.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.logger.info("All OCR results saved: %s", path)
        except OSError:
            self.logger.exception("Could not save OCR debug results")

    @staticmethod
    def confidence_level(confidence: float) -> str:
        """Classify confidence for diagnostics and Gemini restoration guidance."""
        if confidence >= 0.80:
            return "trusted"
        if confidence >= 0.60:
            return "medium"
        if confidence >= 0.40:
            return "low"
        return "very_low"

    @staticmethod
    def _character_count(text: str) -> int:
        """Return visible OCR characters excluding whitespace."""
        return len(re.sub(r"\s+", "", text))

    @staticmethod
    def _low_words(items: list[OCRItem]) -> list[str]:
        """Extract low-confidence word candidates for the AI correction prompt."""
        return [word for item in items if item.confidence <= 0.40 for word in item.text.split()]

    def _merge_words(self, results: list[OCRResult], selected: OCRResult) -> tuple[str, float, list[str]]:
        """Choose the highest-confidence token at each word position across OCR runs."""
        token_sets = [self._token_confidences(result) for result in results]
        selected_tokens = self._token_confidences(selected)
        merged: list[tuple[str, float]] = []
        for index in range(len(selected_tokens)):
            candidates = [tokens[index] for tokens in token_sets if index < len(tokens) and tokens[index][0]]
            merged.append(max(candidates, key=lambda item: item[1]) if candidates else selected_tokens[index])
        text = " ".join(word for word, _ in merged).strip() or selected.text
        confidence = sum(value for _, value in merged) / len(merged) if merged else selected.average_confidence
        return text, confidence, [word for word, value in merged if value <= 0.40]

    @staticmethod
    def _token_confidences(result: OCRResult) -> list[tuple[str, float]]:
        """Expand region confidence into word-confidence pairs for ensemble comparison."""
        return [(word, item.confidence) for item in result.items for word in item.text.split()]

    def is_acceptable(self, result: OCRResult) -> bool:
        """Require the selected/merged OCR result to meet quality thresholds."""
        return (result.average_confidence >= self.config.min_ocr_confidence
                and self._character_count(result.text) >= self.config.min_ocr_characters)

    def _log_first_response(self, raw: Any) -> None:
        """Log EasyOCR's installed version and dynamic response shape once."""
        if self._debug_result_logged:
            return
        import easyocr
        first = raw if self._is_result_tuple(raw) else (raw[0] if isinstance(raw, (list, tuple)) and raw else None)
        self.logger.info("EasyOCR: %s", getattr(easyocr, "__version__", "unknown"))
        self.logger.info("Return type: %s", type(raw).__name__)
        self.logger.info("First item: %r", first)
        self._debug_result_logged = True

    @classmethod
    def _as_items(cls, raw: Any) -> list[Any]:
        """Convert empty, single, or iterable EasyOCR responses into item list."""
        if raw is None:
            return []
        if isinstance(raw, str) or cls._is_result_tuple(raw):
            return [raw]
        try:
            return list(raw)
        except TypeError:
            return [raw]

    @staticmethod
    def _is_box(value: Any) -> bool:
        """Return whether a value resembles an EasyOCR bounding polygon."""
        return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, str)

    @classmethod
    def _is_result_tuple(cls, value: Any) -> bool:
        """Return whether a tuple represents one OCR result rather than a result list."""
        return isinstance(value, tuple) and 1 <= len(value) <= 3 and (isinstance(value[0], str) or cls._is_box(value[0]))

    @classmethod
    def _normalize_item(cls, item: Any) -> OCRItem | None:
        """Normalize all supported EasyOCR result formats without fixed unpacking."""
        box: list[list[int]] | None = None
        text: Any = ""
        confidence: Any = 1.0
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
        if not isinstance(text, str):
            return None
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        return OCRItem(box, text.strip(), confidence)

    @staticmethod
    def _normalize_box(value: Any) -> list[list[int]] | None:
        """Convert an EasyOCR polygon to integer coordinates when available."""
        try:
            return [[int(coordinate) for coordinate in point] for point in value]
        except (TypeError, ValueError):
            return None
