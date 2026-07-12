"""Interruptible local text-to-speech service."""
from __future__ import annotations

import logging
import threading

import cv2

from config import Config


class TTSPlayer:
    """pyttsx3 wrapper with ESC cancellation while speech is in progress."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger
        self.engine: object | None = None

    def initialize(self) -> bool:
        """Initialize pyttsx3 and choose a matching voice where available."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.config.tts_rate)
            voices = self.engine.getProperty("voices")
            target = self.config.tts_voice_gender.lower()
            for voice in voices:
                marker = f"{voice.name} {getattr(voice, 'gender', '')}".lower()
                if target != "default" and target in marker:
                    self.engine.setProperty("voice", voice.id)
                    break
            return True
        except Exception:
            self.logger.exception("TTS initialization failed")
            return False

    def speak(self, text: str) -> None:
        """Speak text without crashing the capture loop; ESC stops it immediately."""
        if self.engine is None or not text:
            return
        error: list[Exception] = []
        def run() -> None:
            try:
                self.engine.say(text)  # type: ignore[union-attr]
                self.engine.runAndWait()  # type: ignore[union-attr]
            except Exception as exc:
                error.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while worker.is_alive():
            if (cv2.waitKey(50) & 0xFF) == 27:
                try:
                    self.engine.stop()  # type: ignore[union-attr]
                except Exception:
                    self.logger.exception("TTS stop failed")
                break
        worker.join(timeout=1)
        if error:
            self.logger.error("TTS failed: %s", error[0])
