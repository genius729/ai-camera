"""Gemini-based multi-result OCR restoration with output validation."""
from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher

from config import Config
from ocr import OCRResult
from performance import PerformanceSession


SYSTEM_RULES = """당신은 OCR 문서 복원 전문가입니다.
입력은 같은 문서를 여러 전처리 방식으로 OCR한 결과입니다. 각 결과에는 일부 오류가 있을 수 있습니다.
confidence, bounding box, 전처리 방법과 문맥을 비교하여 사진 속 원문을 최대한 정확하게 복원하십시오.

규칙:
1. OCR 오인식을 수정합니다.
2. 모든 OCR 결과를 비교합니다.
3. confidence가 높은 결과를 우선 사용합니다.
4. confidence 0.80 이상은 신뢰하고, 0.60~0.80은 보통, 0.60 미만은 낮음, 0.40 미만은 매우 낮음으로 봅니다.
5. 낮은 confidence 단어만 적극적으로 수정하고 높은 confidence 문구는 보존합니다.
6. 맞춤법과 띄어쓰기를 수정하고 잘못된 줄바꿈을 제거합니다.
7. URL, 이메일, 전화번호, 영문 철자와 숫자 형식을 여러 결과를 비교하여 복원합니다.
8. 원문의 의미, 문체, 순서와 사실을 절대로 변경하지 않습니다.
9. 입력에 없는 내용을 추가하거나 요약하지 않습니다.
10. 확신이 없으면 선택된 OCR 원문을 유지합니다.
11. 설명, 제목, 마크다운 없이 수정된 최종 문장만 출력합니다.
"""


class AIProcessor:
    """Restore OCR using all variants and reject unsafe model rewrites."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger = config, logger

    def correct(self, result: OCRResult, performance: PerformanceSession | None = None) -> str:
        """Restore OCR with Gemini and validate the response against selected OCR text."""
        fallback = self._normalize(result.text)
        if not self.config.use_ai or not fallback:
            return fallback
        if performance:
            with performance.measure("AI Postprocess"):
                prompt = self._build_prompt(result)
        else:
            prompt = self._build_prompt(result)
        self.logger.info("Gemini input: %s", prompt)
        try:
            if self.config.ai_provider == "gemini" and self.config.gemini_api_key:
                if performance:
                    with performance.measure("Gemini API"):
                        candidate = self._gemini(prompt)
                else:
                    candidate = self._gemini(prompt)
            elif self.config.openai_api_key:
                candidate = self._openai(prompt)
            else:
                self.logger.warning("No Gemini API key configured; using normalized OCR")
                return fallback
            if performance:
                with performance.measure("AI Postprocess"):
                    candidate = self._normalize(candidate)
                    valid = self._is_valid_response(fallback, candidate)
            else:
                candidate = self._normalize(candidate)
                valid = self._is_valid_response(fallback, candidate)
            self.logger.info("Gemini output: %s", candidate)
            if not valid:
                self.logger.warning("Gemini output rejected by restoration guard; using OCR source")
                return fallback
            return candidate
        except Exception:
            self.logger.exception("Gemini post-processing failed; using normalized OCR")
            return fallback

    def _gemini(self, prompt: str) -> str:
        """Request deterministic restoration through the Gemini SDK."""
        import google.generativeai as genai
        genai.configure(api_key=self.config.gemini_api_key)
        response = genai.GenerativeModel(self.config.gemini_model).generate_content(
            prompt,
            generation_config={"temperature": 0.0},
            request_options={"timeout": 30},
        )
        return response.text or ""

    def _openai(self, prompt: str) -> str:
        """Retain the configured OpenAI provider as a compatible fallback."""
        from openai import OpenAI
        response = OpenAI(api_key=self.config.openai_api_key).chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _build_prompt(result: OCRResult) -> str:
        """Serialize every OCR variant, region confidence, box, and method for Gemini."""
        variants: list[dict[str, object]] = []
        for method, items in result.variants.items():
            variants.append({
                "preprocessing": method,
                "average_confidence": result.variant_scores.get(method, (0.0, 0, 0))[0],
                "regions": [{
                    "text": item.text,
                    "confidence": item.confidence,
                    "confidence_level": AIProcessor._confidence_level(item.confidence),
                    "bounding_box": item.box,
                } for item in items],
            })
        evidence = {
            "selected_preprocessing": result.method,
            "selected_merged_ocr": result.text,
            "selected_average_confidence": result.average_confidence,
            "very_low_confidence_words": result.low_confidence_words,
            "all_ocr_results": variants,
        }
        return SYSTEM_RULES + "\nOCR 증거(JSON):\n" + json.dumps(evidence, ensure_ascii=False, indent=2)

    def _is_valid_response(self, source: str, candidate: str) -> bool:
        """Reject empty, wildly resized, or excessively rewritten Gemini responses."""
        source_key, candidate_key = self._comparison_text(source), self._comparison_text(candidate)
        if not candidate_key or not source_key:
            return False
        length_ratio = len(candidate_key) / len(source_key)
        similarity = SequenceMatcher(None, source_key, candidate_key).ratio()
        change_ratio = 1.0 - similarity
        self.logger.info("Gemini validation: similarity=%.3f, change_ratio=%.3f, length_ratio=%.3f",
                         similarity, change_ratio, length_ratio)
        return (self.config.ai_min_length_ratio <= length_ratio <= self.config.ai_max_length_ratio
                and change_ratio <= self.config.ai_max_change_ratio)

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        """Map numeric confidence to the requested restoration category."""
        if confidence >= 0.80:
            return "trusted"
        if confidence >= 0.60:
            return "medium"
        if confidence >= 0.40:
            return "low"
        return "very_low"

    @staticmethod
    def _comparison_text(text: str) -> str:
        """Normalize spacing only for source/candidate similarity measurement."""
        return re.sub(r"\s+", "", text).lower()

    @staticmethod
    def _normalize(text: str) -> str:
        """Remove wrappers and normalize whitespace without altering document content."""
        text = text.strip().strip("`")
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n", text).strip()
