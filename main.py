"""Entry point for the camera-based reading assistant."""
from __future__ import annotations

import cv2
import numpy as np

from ai_postprocess import AIProcessor
from camera import Camera
from config import CONFIG
from ocr import OCRResult, OCRService
from preprocess import ImagePreprocessor
from tts import TTSPlayer
from utils import ensure_directories, setup_logging, timed


def draw_boxes(frame: np.ndarray, result: OCRResult) -> np.ndarray:
    """Draw bounding boxes from the selected OCR variant on a frame copy."""
    output = frame.copy()
    for item in result.items:
        if item.box is not None:
            cv2.polylines(output, [np.asarray(item.box, dtype=np.int32)], True, (0, 255, 0), 2)
    return output


def show_result(frame: np.ndarray, result: OCRResult, message: str) -> None:
    """Show OCR boxes plus a right-side panel with score and selected method."""
    image = draw_boxes(frame, result)
    panel = np.full((image.shape[0], 400, 3), 25, dtype=np.uint8)
    lines = [message, f"Confidence: {result.average_confidence:.0%}", f"Selected: {result.method or '-'}"]
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (20, 55 + index * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
    cv2.imshow("Reading Assistant", cv2.hconcat([image, panel]))
    cv2.waitKey(700)


def run() -> None:
    """Run the capture, multi-variant OCR, AI correction, and TTS loop."""
    ensure_directories(CONFIG)
    logger = setup_logging(CONFIG)
    camera, tts = Camera(CONFIG, logger), TTSPlayer(CONFIG, logger)
    ocr, processor, ai = OCRService(CONFIG, logger), ImagePreprocessor(CONFIG, logger), AIProcessor(CONFIG, logger)
    if not camera.open():
        logger.error("Camera initialization failed. Check camera connection and index.")
        return
    tts.initialize()
    if not ocr.initialize():
        camera.close()
        return
    try:
        while (frame := camera.wait_for_capture()) is not None:
            if camera.save_capture(frame) is None:
                tts.speak("사진 저장에 실패했습니다. 다시 촬영해주세요.")
                continue
            try:
                with timed(logger, "Preprocessing"):
                    variants = processor.process(frame)
                if CONFIG.debug_mode:
                    logger.debug("OCR input image: %s", CONFIG.debug_dir / "final_ocr_input.jpg")
                if not ocr.has_text(variants["Gray"]):
                    tts.speak("텍스트를 찾지 못했습니다.")
                    continue
                with timed(logger, "Eight-variant OCR and merge"):
                    result = ocr.recognize_variants(variants)
                if CONFIG.debug_mode:
                    logger.debug("OCR merged text: %s", result.text)
                    logger.debug("OCR selected confidence: %.3f", result.average_confidence)
                    logger.debug("OCR selected method: %s", result.method)
                    logger.debug("OCR variant scores: %s", result.variant_scores)
                if not ocr.is_acceptable(result):
                    show_result(frame, result, "OCR FAILED - CAPTURE AGAIN")
                    tts.speak("다시 촬영해주세요.")
                    continue
                show_result(frame, result, "OCR SUCCESS")
                with timed(logger, "Gemini AI post-processing"):
                    corrected = ai.correct(result.text, result.low_confidence_words)
                if CONFIG.debug_mode:
                    logger.debug("AI post-processed text: %s", corrected)
                    logger.debug("TTS final text: %s", corrected)
                tts.speak(corrected)
            except Exception:
                logger.exception("Capture processing failed")
                tts.speak("처리 중 오류가 발생했습니다. 다시 촬영해주세요.")
    finally:
        camera.close()


if __name__ == "__main__":
    run()
