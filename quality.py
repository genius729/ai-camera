"""Pre-OCR camera capture quality assessment."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass(frozen=True)
class QualityResult:
    """Measured capture metrics and the first actionable user guidance."""

    passed: bool
    blur_score: float
    brightness: float
    document_area_ratio: float
    perspective_angle: float
    text_size: float
    message: str = ""

    @property
    def status(self) -> str:
        """Return a concise status suitable for the OpenCV overlay."""
        return "QUALITY OK" if self.passed else "QUALITY RETAKE"


class QualityAssessor:
    """Reject blurred, badly exposed, distant, or skewed captures before OCR."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger

    def assess(self, image: np.ndarray) -> QualityResult:
        """Measure all quality metrics and return prioritized spoken guidance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        document_ratio, perspective = self._document_geometry(gray)
        text_size, component_count = self._estimate_text_size(gray)

        message = ""
        if blur < self.config.blur_min_variance:
            message = "사진이 흐립니다. 조금 더 가까이에서 다시 촬영해주세요."
        elif brightness < self.config.brightness_min:
            message = "주변이 너무 어둡습니다."
        elif brightness > self.config.brightness_max:
            message = "빛 반사가 있습니다."
        elif document_ratio < self.config.document_min_area_ratio:
            message = "조금 더 가까이 촬영해주세요."
        elif perspective > self.config.perspective_max_angle_deviation:
            message = "문서를 정면에서 촬영해주세요."
        elif component_count >= self.config.quality_min_component_count and text_size < self.config.text_min_median_height:
            message = "글자가 너무 작습니다. 조금 더 가까이 촬영해주세요."

        result = QualityResult(not message, blur, brightness, document_ratio, perspective, text_size, message)
        self.logger.info(
            "Capture quality: passed=%s, blur=%.2f, brightness=%.2f, document_size=%.3f, "
            "perspective_angle=%.2f, text_size=%.2f, guidance=%s",
            result.passed, blur, brightness, document_ratio, perspective, text_size, message or "none",
        )
        return result

    @staticmethod
    def _document_geometry(gray: np.ndarray) -> tuple[float, float]:
        """Estimate document coverage and maximum corner deviation from 90 degrees."""
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(gray.shape[0] * gray.shape[1])
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.05:
                break
            polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(polygon) == 4:
                points = QualityAssessor._order_points(polygon.reshape(4, 2).astype(np.float32))
                deviations = [abs(90.0 - QualityAssessor._corner_angle(points[(i - 1) % 4], points[i],
                                                                          points[(i + 1) % 4])) for i in range(4)]
                return area / frame_area, max(deviations)
        # A borderless/full-frame page cannot be measured reliably; do not reject it solely for this.
        return 1.0, 0.0

    @staticmethod
    def _estimate_text_size(gray: np.ndarray) -> tuple[float, int]:
        """Estimate glyph height using plausible connected components before OCR."""
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        heights = [int(row[cv2.CC_STAT_HEIGHT]) for row in stats[1:]
                   if 3 <= int(row[cv2.CC_STAT_HEIGHT]) <= gray.shape[0] * 0.20
                   and 2 <= int(row[cv2.CC_STAT_WIDTH]) <= gray.shape[1] * 0.20
                   and 8 <= int(row[cv2.CC_STAT_AREA]) <= 5000]
        return (float(np.median(heights)), len(heights)) if heights else (0.0, 0)

    @staticmethod
    def _corner_angle(first: np.ndarray, vertex: np.ndarray, third: np.ndarray) -> float:
        """Calculate the angle between two corner vectors in degrees."""
        left, right = first - vertex, third - vertex
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 90.0
        cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        """Order quadrilateral points as top-left, top-right, bottom-right, bottom-left."""
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
        return ordered
