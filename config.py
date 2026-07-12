"""Application configuration for the reading assistant."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load project-local secrets before reading environment variables. Do not commit .env.
load_dotenv(Path(__file__).resolve().parent / ".env")


# Enable detailed OCR/AI/TTS diagnostics with DEBUG_MODE=true in .env.  # Modified
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    """Central, immutable configuration values."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    capture_dir: Path = field(init=False)
    debug_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    min_text_regions: int = 4
    min_ocr_confidence: float = 0.50
    min_ocr_characters: int = 20
    ocr_languages: tuple[str, ...] = ("ko", "en")
    ocr_gpu: bool = False
    debug_mode: bool = DEBUG_MODE  # Modified: controls diagnostic log output.
    tts_rate: int = 165
    tts_voice_gender: str = "female"  # "female", "male", or "default"
    use_ai: bool = True
    ai_provider: str = os.getenv("AI_PROVIDER", "openai").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.0-flash")

    def __post_init__(self) -> None:
        """Derive data folders from the project root."""
        object.__setattr__(self, "capture_dir", self.root_dir / "captures")
        object.__setattr__(self, "debug_dir", self.root_dir / "debug")
        object.__setattr__(self, "log_dir", self.root_dir / "logs")


CONFIG = Config()
