"""Document detection and multi-variant OCR preprocessing."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import Config
from utils import save_debug_image


class ImagePreprocessor:
    """Detect a document, rectify it, and create complementary OCR inputs."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger

    def process(self, source: np.ndarray) -> dict[str, np.ndarray]:
        """Return eight deskewed OCR variants, saving all diagnostic images."""
        document = self._detect_and_rectify(source)
        save_debug_image(self.config.debug_dir / "01_document_detect.jpg", document, self.logger)
        enlarged = self._upscale(document)
        deskewed = self._deskew(enlarged)
        gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        bilateral = cv2.bilateralFilter(clahe, 9, 70, 70)
        clean = self._morphology(bilateral)
        adaptive = cv2.adaptiveThreshold(clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 11)
        otsu = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        sharpen = cv2.filter2D(clean, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        sharp_threshold = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants = {
            "Original": deskewed,
            "Gray": gray,
            "CLAHE": clahe,
            "CLAHE + Bilateral": bilateral,
            "CLAHE + Bilateral + Adaptive Threshold": adaptive,
            "CLAHE + Bilateral + Otsu Threshold": otsu,
            "Sharpen": sharpen,
            "Sharpen + Threshold": sharp_threshold,
        }
        for index, (name, image) in enumerate(variants.items(), start=1):
            safe_name = name.lower().replace(" + ", "_").replace(" ", "_")
            save_debug_image(self.config.debug_dir / f"{index + 1:02d}_{safe_name}.jpg", image, self.logger)
        save_debug_image(self.config.debug_dir / "final_ocr_input.jpg", gray, self.logger)
        return variants

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        """Use an available DNN super-resolution model, else cubic 3x enlargement."""
        model = self.config.root_dir / self.config.superres_model_path
        if hasattr(cv2, "dnn_superres") and model.is_file():
            try:
                superres = cv2.dnn_superres.DnnSuperResImpl_create()
                superres.readModel(str(model))
                superres.setModel("fsrcnn", self.config.ocr_upscale_factor)
                return superres.upsample(image)
            except cv2.error:
                self.logger.warning("Super-resolution unavailable; using INTER_CUBIC", exc_info=True)
        return cv2.resize(image, None, fx=self.config.ocr_upscale_factor,
                          fy=self.config.ocr_upscale_factor, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _detect_and_rectify(image: np.ndarray) -> np.ndarray:
        """Find the largest quadrilateral and apply perspective rectification."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = image.shape[0] * image.shape[1] * 0.15
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < minimum_area:
                break
            corners = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(corners) == 4:
                return ImagePreprocessor._four_point_transform(image, corners.reshape(4, 2).astype(np.float32))
        return image

    @staticmethod
    def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Warp a four-corner document image to a front-facing rectangle."""
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
        top_left, top_right, bottom_right, bottom_left = ordered
        width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
        height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
        if width < 2 or height < 2:
            return image
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        return cv2.warpPerspective(image, cv2.getPerspectiveTransform(ordered, target), (width, height))

    @staticmethod
    def _morphology(image: np.ndarray) -> np.ndarray:
        """Connect character strokes then remove small isolated noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Estimate text angle with minAreaRect and rotate it to level."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coordinates = np.column_stack(np.where(foreground > 0))
        if len(coordinates) < 50:
            return image
        angle = cv2.minAreaRect(coordinates[:, ::-1].astype(np.float32))[2]
        angle = -(90 + angle) if angle < -45 else -angle
        matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
