"""Gemini-first OCR correction with safe offline fallback."""
from __future__ import annotations

import logging
import re

from config import Config


PROMPT = """다음은 카메라 문서 OCR 결과입니다. 원래 의미와 사실을 절대로 바꾸지 말고 읽기 좋은 문장만 출력하세요.

수행할 작업:
- OCR 오인식, 한글/영문 띄어쓰기와 맞춤법, 영문 오타 및 숫자 오류를 문맥으로 보정합니다.
- 줄바꿈을 자연스러운 문단으로 연결하고 불필요한 특수문자와 반복 문자를 제거합니다.
- 문맥을 복원하되, 확신할 수 없는 정보는 추측하여 새로 만들지 않습니다.
- confidence가 매우 낮은 단어는 주변 문맥을 우선 활용해 복원합니다.

confidence가 0.40 이하인 단어:
{low_confidence_words}

OCR 결과:
{ocr_text}
"""


class AIProcessor:
    """Provider-independent OCR correction service, preferring Gemini by configuration."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger

    def correct(self, text: str, low_confidence_words: list[str] | None = None) -> str:
        """Correct OCR with AI, returning conservative cleanup when a call fails."""
        fallback = self._normalize(text)
        if not self.config.use_ai:
            return fallback
        words = low_confidence_words or []
        try:
            if self.config.ai_provider == "gemini" and self.config.gemini_api_key:
                result = self._gemini(text, words)
            elif self.config.openai_api_key:
                result = self._openai(text, words)
            else:
                self.logger.warning("No AI API key configured; using normalized OCR")
                return fallback
            self.logger.info("AI response: %s", result)
            return result.strip() or fallback
        except Exception:
            self.logger.exception("AI post-processing failed; using normalized OCR")
            return fallback

    def _openai(self, text: str, low_confidence_words: list[str]) -> str:
        """Request correction through the official OpenAI Python SDK."""
        from openai import OpenAI
        response = OpenAI(api_key=self.config.openai_api_key).chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": self._build_prompt(text, low_confidence_words)}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _gemini(self, text: str, low_confidence_words: list[str]) -> str:
        """Request correction through the Google Generative AI SDK."""
        import google.generativeai as genai
        genai.configure(api_key=self.config.gemini_api_key)
        response = genai.GenerativeModel(self.config.gemini_model).generate_content(
            self._build_prompt(text, low_confidence_words)
        )
        return response.text

    @staticmethod
    def _build_prompt(text: str, low_confidence_words: list[str]) -> str:
        """Build the correction prompt with low-confidence OCR evidence."""
        return PROMPT.format(low_confidence_words=", ".join(low_confidence_words) or "없음", ocr_text=text)

    @staticmethod
    def _normalize(text: str) -> str:
        """Apply conservative cleanup when an online AI provider is unavailable."""
        text = re.sub(r"[|_~*#%]+", " ", text)
        text = re.sub(r"(.)\1{5,}", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n", text).strip()
