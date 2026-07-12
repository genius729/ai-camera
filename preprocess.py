"""Document-oriented image preprocessing pipeline."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import Config
from utils import save_debug_image


class ImagePreprocessor:
    """Create OCR-ready, deskewed document images and debug artifacts."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger

    def process(self, source: np.ndarray) -> np.ndarray:
        """Run the required OCR enhancement stages and return final grayscale image."""
        d = self.config.debug_dir
        save_debug_image(d / "01_original.jpg", source, self.logger)
        resized = self._resize(source)
        save_debug_image(d / "02_resize.jpg", resized, self.logger)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        save_debug_image(d / "03_gray.jpg", gray, self.logger)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        save_debug_image(d / "04_clahe.jpg", clahe, self.logger)
        filtered = cv2.bilateralFilter(clahe, 9, 70, 70)
        save_debug_image(d / "05_bilateral.jpg", filtered, self.logger)
        threshold = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 11)
        save_debug_image(d / "06_threshold.jpg", threshold, self.logger)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        save_debug_image(d / "07_morphology.jpg", morph, self.logger)
        cropped = self._crop_document(resized)
        save_debug_image(d / "08_crop.jpg", cropped, self.logger)
        deskewed = self._deskew(cropped)
        save_debug_image(d / "09_deskew.jpg", deskewed, self.logger)
        sharp = cv2.filter2D(deskewed, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        save_debug_image(d / "10_sharpen.jpg", sharp, self.logger)
        final = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        save_debug_image(d / "11_final_ocr_input.jpg", final, self.logger)
        save_debug_image(d / "final_ocr_input.jpg", final, self.logger)
        return final

    @staticmethod
    def _resize(image: np.ndarray) -> np.ndarray:
        """Upscale small images for more readable OCR glyphs."""
        height, width = image.shape[:2]
        scale = max(1.0, min(2.0, 1600 / max(height, width)))
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _crop_document(image: np.ndarray) -> np.ndarray:
        """Crop the largest plausible quadrilateral document, otherwise keep source."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = image.shape[0] * image.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < image_area * 0.15:
                break
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                return image[y:y + h, x:x + w]
        return image

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Deskew text using the minimum-area rectangle of foreground pixels."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0))
        if len(coords) < 50:
            return image
        angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[2]
        angle = -(90 + angle) if angle < -45 else -angle
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
