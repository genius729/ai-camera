"""TTS backends intended to run exclusively in a dedicated worker thread."""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import tempfile
import threading
import uuid  # Modified: create a unique Windows MCI playback alias.
from pathlib import Path
from typing import Any

from config import Config


class TTSPlayer:
    """Use Edge-TTS first with a same-thread pyttsx3 offline fallback."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger
        self._engine: Any | None = None
        self._owner_thread: int | None = None
        self._stop_requested = threading.Event()

    def initialize(self) -> bool:
        """Record ownership; pyttsx3 is initialized lazily in this same thread."""
        self._owner_thread = threading.get_ident()
        self.logger.info("TTS worker initialized: backend=%s, thread=%s",
                         self.config.tts_backend, self._owner_thread)
        return True

    def speak(self, text: str) -> None:
        """Speak synchronously in the TTS worker without calling any OpenCV API."""
        if not text or self._stop_requested.is_set():
            return
        if self._owner_thread != threading.get_ident():
            raise RuntimeError("TTSPlayer.speak() must run on its owning TTS thread")
        if self.config.tts_backend == "edge":
            try:
                self._speak_edge(text)
                return
            except Exception:
                self.logger.exception("Edge-TTS failed; falling back to local pyttsx3")
        self._speak_pyttsx3(text)

    def stop(self) -> None:
        """Request cancellation and stop active Windows audio playback."""
        self._stop_requested.set()
        # pyttsx3/Windows COM methods must only be called by the owning TTS thread.
        if self._engine is not None and self._owner_thread == threading.get_ident():
            try:
                self._engine.stop()
            except Exception:
                self.logger.exception("Could not stop pyttsx3")
        if os.name == "nt":
            try:
                ctypes.windll.winmm.mciSendStringW("close all", None, 0, None)
            except Exception:
                self.logger.exception("Could not stop Edge-TTS playback")

    def reset_stop(self) -> None:
        """Clear cancellation before the worker accepts another utterance."""
        self._stop_requested.clear()

    def close(self) -> None:
        """Stop playback and release the local engine."""
        self.stop()
        self._engine = None

    def _speak_edge(self, text: str) -> None:
        """Generate an Edge-TTS MP3 and play it in the worker thread."""
        import edge_tts

        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temporary:
                path = Path(temporary.name)
            communicate = edge_tts.Communicate(text, self.config.edge_tts_voice,
                                                rate=self.config.edge_tts_rate)
            asyncio.run(communicate.save(str(path)))
            if not self._stop_requested.is_set():
                self._play_mp3_windows(path)  # Modified: use the built-in Windows API.
        finally:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    self.logger.exception("Could not remove temporary TTS audio: %s", path)

    def _play_mp3_windows(self, path: Path) -> None:
        """Play an MP3 synchronously through the Windows Media Control Interface."""
        # Modified: MCI supports MP3 without a third-party playback dependency.
        if os.name != "nt":
            raise RuntimeError("Edge-TTS MP3 playback requires Windows")

        winmm = ctypes.windll.winmm
        alias = f"edge_tts_{uuid.uuid4().hex}"

        def mci(command: str) -> None:
            error_code = winmm.mciSendStringW(command, None, 0, None)
            if error_code:
                message = ctypes.create_unicode_buffer(256)
                winmm.mciGetErrorStringW(error_code, message, len(message))
                raise RuntimeError(
                    f"Windows MCI error {error_code}: {message.value or command}"
                )

        try:  # Modified: always close the device, including after playback errors.
            mci(f'open "{path.resolve()}" type mpegvideo alias {alias}')
            mci(f"play {alias} wait")
        finally:
            # A failed/interrupting close must not terminate the TTS worker.
            close_error = winmm.mciSendStringW(f"close {alias}", None, 0, None)
            if close_error and not self._stop_requested.is_set():
                self.logger.warning("Could not close Windows TTS audio device: error=%s",
                                    close_error)

    def _speak_pyttsx3(self, text: str) -> None:
        """Run pyttsx3 entirely in one thread to avoid Windows COM conflicts."""
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.config.tts_rate)
            target = self.config.tts_voice_gender.lower()
            for voice in self._engine.getProperty("voices"):
                marker = f"{voice.name} {getattr(voice, 'gender', '')}".lower()
                if target != "default" and target in marker:
                    self._engine.setProperty("voice", voice.id)
                    break
        if not self._stop_requested.is_set():
            self._engine.say(text)
            self._engine.runAndWait()
