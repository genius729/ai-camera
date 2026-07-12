"""Entry point for the camera-based reading assistant."""
from __future__ import annotations

import logging

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
    """Draw OCR text-region boxes onto a display copy of the captured frame."""
    output = frame.copy()
    for item in result.items:
        if item.box is None:
            continue
        points = np.asarray(item.box, dtype=np.int32)
        cv2.polylines(output, [points], True, (0, 255, 0), 2)
    return output


def show_result(frame: np.ndarray, result: OCRResult, message: str) -> None:
    """Show OCR boxes and the current outcome briefly in the OpenCV window."""
    display = draw_boxes(frame, result)
    cv2.putText(display, message, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Reading Assistant", display)
    cv2.waitKey(400)


def run() -> None:
    """Initialize services and run the capture-to-speech loop until ESC."""
    ensure_directories(CONFIG)
    logger = setup_logging(CONFIG)
    camera = Camera(CONFIG, logger)
    tts = TTSPlayer(CONFIG, logger)
    ocr = OCRService(CONFIG, logger)
    processor = ImagePreprocessor(CONFIG, logger)
    ai = AIProcessor(CONFIG, logger)
    if not camera.open():
        logger.error("Camera initialization failed. Check camera connection and index.")
        return
    tts.initialize()  # TTS failure should not prevent visual OCR operation.
    if not ocr.initialize():
        camera.close()
        return
    logger.info("Reading assistant started")
    try:
        while True:
            frame = camera.wait_for_capture()
            if frame is None:
                break
            if camera.save_capture(frame) is None:
                tts.speak("사진 저장에 실패했습니다. 다시 촬영해주세요.")
                continue
            try:
                with timed(logger, "Preprocessing"):
                    final_image = processor.process(frame)
                if CONFIG.debug_mode:  # Modified
                    logger.debug("OCR input image: %s", CONFIG.debug_dir / "final_ocr_input.jpg")
                if not ocr.has_text(final_image):
                    logger.info("No text detected")
                    tts.speak("텍스트를 찾지 못했습니다.")
                    continue
                with timed(logger, "OCR"):
                    result = ocr.recognize(final_image)
                if CONFIG.debug_mode:  # Modified
                    logger.debug("OCR raw text: %s", result.text)
                    logger.debug("OCR average confidence: %.3f", result.average_confidence)
                if not ocr.is_acceptable(result):
                    show_result(frame, result, "Please capture again")
                    tts.speak("다시 촬영해주세요.")
                    continue
                show_result(frame, result, "OCR success")
                with timed(logger, "AI post-processing"):
                    corrected = ai.correct(result.text)
                if CONFIG.debug_mode:  # Modified
                    logger.debug("AI post-processed text: %s", corrected)
                    logger.debug("TTS final text: %s", corrected)
                tts.speak(corrected)
            except Exception:
                logger.exception("Capture processing failed")
                tts.speak("처리 중 오류가 발생했습니다. 다시 촬영해주세요.")
    finally:
        camera.close()
        logger.info("Reading assistant stopped")


if __name__ == "__main__":
    run()
