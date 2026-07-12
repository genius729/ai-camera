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
    blur_min_variance: float = 100.0
    brightness_min: float = 45.0
    brightness_max: float = 220.0
    document_min_area_ratio: float = 0.20
    perspective_max_angle_deviation: float = 18.0
    text_min_median_height: float = 10.0
    quality_min_component_count: int = 4
    ocr_languages: tuple[str, ...] = ("ko", "en")
    ocr_gpu: bool = False
    ocr_upscale_factor: int = 3
    easyocr_canvas_size: int = 2560
    superres_model_path: Path = field(default_factory=lambda: Path("models/FSRCNN_x3.pb"))
    debug_mode: bool = DEBUG_MODE  # Modified: controls diagnostic log output.
    tts_rate: int = 165
    tts_voice_gender: str = "female"  # "female", "male", or "default"
    tts_backend: str = os.getenv("TTS_BACKEND", "edge").lower()
    edge_tts_voice: str = os.getenv("EDGE_TTS_VOICE", "ko-KR-SunHiNeural")
    edge_tts_rate: str = os.getenv("EDGE_TTS_RATE", "+0%")
    use_ai: bool = True
    ai_provider: str = os.getenv("AI_PROVIDER", "gemini").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    ai_max_change_ratio: float = 0.55
    ai_min_length_ratio: float = 0.45
    ai_max_length_ratio: float = 1.60
    worker_queue_size: int = 2

    def __post_init__(self) -> None:
        """Derive data folders from the project root."""
        object.__setattr__(self, "capture_dir", self.root_dir / "captures")
        object.__setattr__(self, "debug_dir", self.root_dir / "debug")
        object.__setattr__(self, "log_dir", self.root_dir / "logs")


CONFIG = Config()
