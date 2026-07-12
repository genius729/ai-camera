"""Optional AI-based OCR cleanup with an offline-safe fallback."""
from __future__ import annotations

import logging
import re

from config import Config


PROMPT = """다음 OCR 결과를 수정하세요. OCR 오인식, 띄어쓰기, 맞춤법을 수정하고 문장이 자연스럽게 이어지도록 하세요.
불필요한 특수문자와 반복 문자를 제거하되 원래 의미는 절대 바꾸지 마세요. 출력은 수정된 문장만 작성하세요.

OCR 결과:
"""


class AIProcessor:
    """Provider-independent OCR correction service."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger

    def correct(self, text: str) -> str:
        """Call configured AI provider; return locally normalized text on any failure."""
        fallback = self._normalize(text)
        if not self.config.use_ai:
            return fallback
        try:
            if self.config.ai_provider == "gemini" and self.config.gemini_api_key:
                result = self._gemini(text)
            elif self.config.openai_api_key:
                result = self._openai(text)
            else:
                self.logger.warning("No AI API key configured; using OCR text without AI correction")
                return fallback
            self.logger.info("AI response: %s", result)
            return result.strip() or fallback
        except Exception:
            self.logger.exception("AI post-processing failed; using normalized OCR")
            return fallback

    def _openai(self, text: str) -> str:
        """Request correction through the official OpenAI Python SDK."""
        from openai import OpenAI
        client = OpenAI(api_key=self.config.openai_api_key)
        response = client.chat.completions.create(model=self.config.openai_model,
            messages=[{"role": "user", "content": PROMPT + text}], temperature=0)
        return response.choices[0].message.content or ""

    def _gemini(self, text: str) -> str:
        """Request correction through the Google Generative AI SDK."""
        import google.generativeai as genai
        genai.configure(api_key=self.config.gemini_api_key)
        response = genai.GenerativeModel(self.config.gemini_model).generate_content(PROMPT + text)
        return response.text

    @staticmethod
    def _normalize(text: str) -> str:
        """Apply conservative cleanup when an online AI provider is unavailable."""
        text = re.sub(r"[|_~*#%]+", " ", text)
        text = re.sub(r"(.)\1{5,}", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n", text).strip()
