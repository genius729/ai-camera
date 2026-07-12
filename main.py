"""Responsive main-thread OpenCV loop with queued OCR, AI, and TTS workers."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai_postprocess import AIProcessor
from camera import Camera
from config import CONFIG, Config
from ocr import OCRResult, OCRService
from preprocess import ImagePreprocessor
from performance import PerformanceSession
from quality import QualityAssessor, QualityResult
from tts import TTSPlayer
from utils import ensure_directories, setup_logging, timed


class AppState(str, Enum):
    """Application state machine values shared by UI and workers."""

    PREVIEW = "PREVIEW"
    CAPTURE = "CAPTURE"
    PROCESSING = "PROCESSING"
    OCR = "OCR"
    AI = "AI"
    TTS = "TTS"
    READY = "READY"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CaptureJob:
    """Immutable camera frame handed from the main thread to OCR processing."""

    job_id: int
    frame: np.ndarray
    performance: PerformanceSession


@dataclass(frozen=True)
class AIJob:
    """Complete OCR evidence handed from the OCR worker to the AI worker."""

    job_id: int
    result: OCRResult
    selected_image: np.ndarray
    quality: QualityResult
    performance: PerformanceSession


@dataclass(frozen=True)
class TTSJob:
    """Final or guidance text handed to the dedicated TTS worker."""

    job_id: int
    text: str
    performance: PerformanceSession


@dataclass(frozen=True)
class UISnapshot:
    """Thread-safe immutable view of the current application state."""

    state: AppState
    message: str
    quality: QualityResult | None
    result: OCRResult | None
    selected_image: np.ndarray | None
    captured_frame: np.ndarray | None


class StateMachine:
    """Synchronize state transitions and the latest OCR display data."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = AppState.PREVIEW
        self._message = "촬영 대기"
        self._quality: QualityResult | None = None
        self._result: OCRResult | None = None
        self._selected_image: np.ndarray | None = None
        self._captured_frame: np.ndarray | None = None

    def transition(self, state: AppState, message: str = "") -> None:
        """Atomically change application state and optional user-facing message."""
        with self._lock:
            self._state = state
            self._message = message

    def set_quality(self, quality: QualityResult) -> None:
        """Publish the most recent capture-quality measurements."""
        with self._lock:
            self._quality = quality

    def set_ocr_result(self, result: OCRResult, image: np.ndarray) -> None:
        """Publish OCR output for rendering exclusively by the main thread."""
        with self._lock:
            self._result = result
            self._selected_image = image

    def begin_capture(self, captured_frame: np.ndarray) -> None:
        """Freeze the exact cloned frame and enter CAPTURE state atomically."""
        with self._lock:
            self._captured_frame = captured_frame
            self._quality = None
            self._result = None
            self._selected_image = None
            self._state = AppState.CAPTURE
            self._message = "촬영 완료"

    def release_to_ready(self, message: str = "촬영 준비 완료") -> None:
        """Release the frozen frame and resume live preview in READY state."""
        with self._lock:
            self._captured_frame = None
            self._selected_image = None
            self._result = None
            self._quality = None
            self._state = AppState.READY
            self._message = message

    def snapshot(self) -> UISnapshot:
        """Return a consistent state snapshot without exposing mutable control data."""
        with self._lock:
            return UISnapshot(self._state, self._message, self._quality,
                              self._result, self._selected_image, self._captured_frame)

    def accepts_capture(self) -> bool:
        """Return whether SPACE may enqueue a new capture."""
        with self._lock:
            return self._state in {AppState.PREVIEW, AppState.READY}


class AsyncReadingAssistant:
    """Coordinate main-thread preview with independent queued worker stages."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger
        self.camera = Camera(config, logger)
        self.ocr = OCRService(config, logger)
        self.processor = ImagePreprocessor(config, logger)
        self.quality = QualityAssessor(config, logger)
        self.ai = AIProcessor(config, logger)
        self.state = StateMachine()
        self.stop_event = threading.Event()
        self.capture_queue: queue.Queue[CaptureJob] = queue.Queue(config.worker_queue_size)
        self.ai_queue: queue.Queue[AIJob] = queue.Queue(config.worker_queue_size)
        self.tts_queue: queue.Queue[TTSJob] = queue.Queue(config.worker_queue_size)
        self.tts_player: TTSPlayer | None = None
        self.threads: list[threading.Thread] = []
        self._banner_cache: dict[tuple[str, int], np.ndarray] = {}
        self._job_lock = threading.RLock()
        self._next_job_id = 0
        self._active_job_id: int | None = None
        self._cancelled_jobs: set[int] = set()

    def run(self) -> None:
        """Run the OpenCV preview and Windows message pump only on the main thread."""
        if not self.camera.open():
            self.logger.error("Camera initialization failed")
            return
        self._start_workers()
        try:
            while not self.stop_event.is_set():
                frame = self.camera.read_frame()
                if frame is not None:
                    cv2.imshow("Reading Assistant", self._render_preview(frame, self.state.snapshot()))
                # This is the only cv2.waitKey call in the application runtime.
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    if self.state.accepts_capture():
                        break
                    self._cancel_active_job()
                    continue
                if key == ord(" ") and frame is not None:
                    self._enqueue_capture(frame)
        finally:
            self.shutdown()

    def _start_workers(self) -> None:
        """Start persistent OCR, AI, and TTS queue consumers."""
        targets = (("ocr-worker", self._ocr_worker), ("ai-worker", self._ai_worker),
                   ("tts-worker", self._tts_worker))
        self.threads = [threading.Thread(name=name, target=target, daemon=True)
                        for name, target in targets]
        for thread in self.threads:
            thread.start()
        self.logger.info("Worker threads started: %s", [thread.name for thread in self.threads])

    def _enqueue_capture(self, frame: np.ndarray) -> None:
        """Clone once and share that exact frame with both Freeze UI and OCR."""
        if not self.state.accepts_capture():
            return
        with self._job_lock:
            self._next_job_id += 1
            job_id = self._next_job_id
            self._active_job_id = job_id
        performance = PerformanceSession(job_id, self.logger)
        with performance.measure("Camera Capture"):
            captured_frame = frame.copy()
        try:
            self.capture_queue.put_nowait(CaptureJob(job_id, captured_frame, performance))
            self.state.begin_capture(captured_frame)
        except queue.Full:
            with self._job_lock:
                self._active_job_id = None
            self.state.transition(AppState.ERROR, "처리 대기열이 가득 찼습니다")

    def _cancel_active_job(self) -> None:
        """Cancel the frozen capture pipeline and immediately resume READY preview."""
        with self._job_lock:
            if self._active_job_id is not None:
                self._cancelled_jobs.add(self._active_job_id)
                self.logger.info("Cancelled capture job: %s", self._active_job_id)
                self._active_job_id = None
        if self.tts_player is not None:
            self.tts_player.stop()
        self.state.release_to_ready("작업이 취소되었습니다")

    def _is_cancelled(self, job_id: int) -> bool:
        """Return whether ESC cancelled a specific capture generation."""
        with self._job_lock:
            return job_id in self._cancelled_jobs

    def _complete_job(self, job_id: int) -> None:
        """Release Freeze only when the active capture's TTS stage has completed."""
        with self._job_lock:
            if self._active_job_id != job_id:
                return
            self._active_job_id = None
            self._cancelled_jobs.discard(job_id)
        self.state.release_to_ready()

    def _ocr_worker(self) -> None:
        """Consume captures and perform quality assessment, preprocessing, and EasyOCR."""
        # Begin the expensive model load at application startup so the first
        # shutter press does not pay the complete cold-start cost.
        ocr_initialized = self.ocr.initialize()
        while not self.stop_event.is_set():
            try:
                job = self.capture_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self._is_cancelled(job.job_id):
                    continue
                self.state.transition(AppState.PROCESSING, "촬영 품질 검사 중")
                if not ocr_initialized:
                    self.state.transition(AppState.PROCESSING, "문자 인식 모델 초기화 중")
                    ocr_initialized = self.ocr.initialize()
                    if not ocr_initialized:
                        self.state.transition(AppState.ERROR, "문자 인식 초기화 오류")
                        self._queue_guidance(job.job_id, "문자 인식 기능을 초기화하지 못했습니다. 다시 시도해주세요.")
                        continue
                if self._is_cancelled(job.job_id):
                    continue
                with job.performance.measure("Camera Capture"):
                    saved_capture = self.camera.save_capture(job.frame)
                if saved_capture is None:
                    self._queue_guidance(job.job_id, "사진 저장에 실패했습니다. 다시 촬영해주세요.")
                    continue
                with job.performance.measure("Quality Check"):
                    quality = self.quality.assess(job.frame)
                self.state.set_quality(quality)
                if self._is_cancelled(job.job_id):
                    continue
                if not quality.passed:
                    self.state.transition(AppState.ERROR, quality.message)
                    self._queue_guidance(job.job_id, quality.message)
                    continue
                variants = self.processor.process(job.frame, job.performance)
                if self._is_cancelled(job.job_id):
                    continue
                if not self.ocr.has_text(variants["Gray"]):
                    self._queue_guidance(job.job_id, "텍스트를 찾지 못했습니다.")
                    continue
                self.state.transition(AppState.OCR, "문자 인식 중")
                result = self.ocr.recognize_variants(variants, job.performance)
                if self._is_cancelled(job.job_id):
                    continue
                selected_image = variants.get(result.method, variants["Gray"])
                self.state.set_ocr_result(result, selected_image)
                if not self.ocr.is_acceptable(result):
                    self._queue_guidance(job.job_id, "다시 촬영해주세요.")
                    continue
                self.ai_queue.put(AIJob(job.job_id, result, selected_image, quality,
                                        job.performance), timeout=1.0)
            except queue.Full:
                self.logger.error("AI work queue is full")
                if not self._is_cancelled(job.job_id):
                    self._queue_guidance(job.job_id, "처리가 지연되고 있습니다. 다시 촬영해주세요.")
            except Exception:
                self.logger.exception("OCR worker job failed")
                if not self._is_cancelled(job.job_id):
                    self.state.transition(AppState.ERROR, "문자 인식 오류")
                    self._queue_guidance(job.job_id, "문자 인식 중 오류가 발생했습니다. 다시 촬영해주세요.")
            finally:
                self.capture_queue.task_done()

    def _ai_worker(self) -> None:
        """Consume complete OCR evidence and call Gemini without blocking preview or OCR UI."""
        while not self.stop_event.is_set():
            try:
                job = self.ai_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self._is_cancelled(job.job_id):
                    continue
                self.state.transition(AppState.AI, "인공지능 문장 복원 중")
                corrected = self.ai.correct(job.result, job.performance)
                if self._is_cancelled(job.job_id):
                    continue
                self.logger.info("Final TTS text: %s", corrected)
                self.tts_queue.put(TTSJob(job.job_id, corrected, job.performance), timeout=1.0)
            except queue.Full:
                self.logger.error("TTS work queue is full")
                if not self._is_cancelled(job.job_id):
                    self.state.transition(AppState.ERROR, "음성 대기열 오류")
            except Exception:
                self.logger.exception("AI worker job failed")
                if not self._is_cancelled(job.job_id):
                    self.state.transition(AppState.ERROR, "AI 처리 오류")
                    self._queue_guidance(job.job_id, "인공지능 처리에 실패했습니다. 인식된 원문을 읽습니다.",
                                         job.result.text)
            finally:
                self.ai_queue.task_done()

    def _tts_worker(self) -> None:
        """Own all TTS/COM resources and consume speech jobs until shutdown."""
        player = TTSPlayer(self.config, self.logger)
        self.tts_player = player
        player.initialize()
        try:
            while not self.stop_event.is_set():
                try:
                    job = self.tts_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if self.stop_event.is_set():
                        continue
                    if self._is_cancelled(job.job_id):
                        continue
                    player.reset_stop()
                    self.state.transition(AppState.TTS, "음성 출력 중")
                    player.speak(job.text, job.performance)
                    report = job.performance.report()
                    self.logger.info("\n%s", report)
                    print(report, flush=True)
                    if not self.stop_event.is_set() and not self._is_cancelled(job.job_id):
                        self._complete_job(job.job_id)
                except Exception:
                    self.logger.exception("TTS worker job failed")
                    if not self._is_cancelled(job.job_id):
                        self.state.transition(AppState.ERROR, "음성 출력 오류")
                        self._complete_job(job.job_id)
                finally:
                    self.tts_queue.task_done()
        finally:
            player.close()

    def _queue_guidance(self, job_id: int, guidance: str, followup: str = "") -> None:
        """Queue spoken guidance, optionally followed by recognized source text."""
        text = f"{guidance} {followup}".strip()
        try:
            if not self._is_cancelled(job_id):
                # Guidance jobs originate in the capture worker and do not have a full
                # successful-pipeline report, so use a new session for TTS diagnostics.
                self.tts_queue.put(TTSJob(job_id, text,
                                          PerformanceSession(job_id, self.logger)), timeout=1.0)
        except queue.Full:
            self.logger.error("Could not enqueue TTS guidance")
            self.state.transition(AppState.ERROR, guidance)

    def _render_preview(self, frame: np.ndarray, snapshot: UISnapshot) -> np.ndarray:
        """Render live video when ready, otherwise the exact frozen captured frame."""
        source = snapshot.captured_frame if snapshot.captured_frame is not None else frame
        live = source.copy()
        banner_text = self._state_banner(snapshot.state)
        banner = self._make_banner(banner_text, live.shape[1])
        live[:banner.shape[0], :banner.shape[1]] = banner
        cv2.putText(live, f"STATE: {snapshot.state.value}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
        panel = self._render_result_panel(live.shape[0], snapshot)
        return cv2.hconcat([live, panel])

    def _render_result_panel(self, height: int, snapshot: UISnapshot) -> np.ndarray:
        """Render latest OCR image, boxes, confidence, method, and quality metrics."""
        panel = np.full((height, 460, 3), 25, dtype=np.uint8)
        lines = self._progress_lines(snapshot.state)
        if snapshot.result is not None:
            lines.extend([f"Confidence: {snapshot.result.average_confidence:.0%}",
                          f"Selected: {snapshot.result.method}"])
        if snapshot.quality is not None:
            lines.extend([f"Quality: {snapshot.quality.status}",
                          f"Blur: {snapshot.quality.blur_score:.1f}",
                          f"Brightness: {snapshot.quality.brightness:.1f}"])
        if snapshot.selected_image is not None and snapshot.result is not None:
            result_image = self._draw_boxes(snapshot.selected_image, snapshot.result)
            available_height = max(1, height - 230)
            scale = min(430 / result_image.shape[1], available_height / result_image.shape[0], 1.0)
            result_image = cv2.resize(result_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            y = height - result_image.shape[0] - 10
            panel[y:y + result_image.shape[0], 15:15 + result_image.shape[1]] = result_image
        try:
            font_path = Path("C:/Windows/Fonts/malgun.ttf")
            font = ImageFont.truetype(str(font_path), 19) if font_path.is_file() else ImageFont.load_default()
            state_font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else font
            pil_panel = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            drawer = ImageDraw.Draw(pil_panel)
            display_state = "CAPTURED" if snapshot.state == AppState.CAPTURE else snapshot.state.value
            drawer.text((315, 12), f"[{display_state}]", font=state_font, fill=(0, 255, 0))
            for index, line in enumerate(lines):
                drawer.text((15, 18 + index * 29), line, font=font, fill=(0, 220, 0))
            panel = cv2.cvtColor(np.asarray(pil_panel), cv2.COLOR_RGB2BGR)
        except Exception:
            self.logger.exception("Could not render result status panel")
        return panel

    @staticmethod
    def _progress_lines(state: AppState) -> list[str]:
        """Return Freeze Preview progress guidance for the current pipeline stage."""
        if state in {AppState.CAPTURE, AppState.PROCESSING}:
            return ["📷 촬영 완료", "이미지 분석 중...", "잠시만 기다려 주세요."]
        if state == AppState.OCR:
            return ["📷 촬영 완료", "OCR 진행 중...", "잠시만 기다려 주세요."]
        if state == AppState.AI:
            return ["OCR 완료", "Gemini 처리 중...", "잠시만 기다려 주세요."]
        if state == AppState.TTS:
            return ["OCR 완료", "Gemini 완료", "음성 출력 중..."]
        if state == AppState.ERROR:
            return ["처리 오류", "음성 안내 후 다시 촬영할 수 있습니다."]
        return ["촬영 준비 완료", "SPACE : 촬영", "ESC : 종료"]

    @staticmethod
    def _draw_boxes(image: np.ndarray, result: OCRResult) -> np.ndarray:
        """Draw OCR boxes on a processed-image copy; called only by the main thread."""
        output = image.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        for item in result.items:
            if item.box is not None:
                cv2.polylines(output, [np.asarray(item.box, dtype=np.int32)], True, (0, 255, 0), 2)
        return output

    @staticmethod
    def _state_banner(state: AppState) -> str:
        """Map state-machine values to concise Korean preview guidance."""
        if state in {AppState.CAPTURE, AppState.PROCESSING, AppState.OCR, AppState.AI}:
            return "처리 중..."
        if state == AppState.TTS:
            return "음성 출력 중..."
        if state == AppState.ERROR:
            return "오류가 발생했습니다. 다시 촬영해주세요."
        return "SPACE : 촬영    ESC : 종료"

    def _make_banner(self, text: str, width: int) -> np.ndarray:
        """Create and cache a Korean-capable status banner using Pillow."""
        key = (text, width)
        if key in self._banner_cache:
            return self._banner_cache[key]
        banner = np.full((58, width, 3), 20, dtype=np.uint8)
        try:
            font_path = Path("C:/Windows/Fonts/malgun.ttf")
            font = ImageFont.truetype(str(font_path), 28) if font_path.is_file() else ImageFont.load_default()
            pil_image = Image.fromarray(cv2.cvtColor(banner, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(pil_image).text((18, 10), text, font=font, fill=(0, 255, 0))
            banner = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            self.logger.exception("Could not render Korean preview banner")
            cv2.putText(banner, "Processing..." if "..." in text else "READY", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self._banner_cache[key] = banner
        return banner

    def shutdown(self) -> None:
        """Signal workers, cancel audio, join threads, and release GUI/camera resources."""
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if self.tts_player is not None:
            self.tts_player.stop()
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.logger.warning("Worker did not stop before timeout: %s", thread.name)
        self.camera.close()
        self.logger.info("Asynchronous reading assistant stopped")


def run() -> None:
    """Configure and launch the asynchronous reading assistant."""
    ensure_directories(CONFIG)
    logger = setup_logging(CONFIG)
    AsyncReadingAssistant(CONFIG, logger).run()


if __name__ == "__main__":
    run()
