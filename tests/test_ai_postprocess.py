"""Run with: python -m unittest discover -s tests -v"""
import json
import logging
import unittest
from dataclasses import replace
from unittest.mock import patch

from ai_postprocess import AIProcessor, SYSTEM_RULES
from config import Config
from ocr import OCRItem, OCRResult


class AIProcessorTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.processor = AIProcessor(self.config, logging.getLogger("test-ai"))

    def test_prompt_includes_alternative_text_and_low_confidence_box(self):
        primary = OCRItem([[0, 0], [80, 0], [80, 15], [0, 15]], "의료비 15,000원", 0.51)
        alternative = OCRItem([[0, 0], [80, 0], [80, 15], [0, 15]], "의료비 15,000원", 0.9)
        extra = OCRItem([[0, 20], [80, 20], [80, 35], [0, 35]], "납부기한", 0.95)
        result = OCRResult(
            items=[primary], text="의료비 15,000원", average_confidence=0.51,
            method="Original", variant_scores={"Gray": (0.92, 12, 14)},
            variants={"Original": [primary], "Gray": [alternative, extra]},
        )
        prompt = self.processor._build_prompt(result)
        evidence = json.loads(prompt.split("OCR 증거(JSON):\n", 1)[1])
        self.assertEqual(evidence["selected_text"], "의료비 15,000원")
        self.assertEqual(evidence["uncertain_regions"][0]["text"], "의료비 15,000원")
        self.assertIn("납부기한", evidence["other_ocr_candidates"][0]["text"])
        self.assertNotIn(SYSTEM_RULES, prompt)

    def test_number_or_url_rewrite_is_rejected(self):
        source = "납부금액은 15,000원입니다. 링크 https://example.com/a"
        self.assertFalse(self.processor._is_valid_response(
            source, "납부금액은 150,000원입니다. 링크 https://example.com/a"))
        self.assertFalse(self.processor._is_valid_response(
            source, "납부금액은 15,000원입니다. 링크 https://example.com/b"))

    def test_safe_spacing_correction_is_accepted(self):
        self.assertTrue(self.processor._is_valid_response(
            "오늘 은 비 가 내립니다.", "오늘은 비가 내립니다."))

    def test_ai_failure_uses_source(self):
        processor = AIProcessor(
            replace(self.config, openai_api_key="fake-key", ai_provider="openai"),
            logging.getLogger("test-ai"),
        )
        result = OCRResult([], "시험 원문 2026년", 0.8)
        with patch.object(processor, "_openai", side_effect=RuntimeError("offline")):
            self.assertEqual(processor.correct(result), "시험 원문 2026년")

    def test_candidate_rejected_returns_source(self):
        processor = AIProcessor(
            replace(self.config, openai_api_key="fake-key", ai_provider="openai"),
            logging.getLogger("test-ai"),
        )
        result = OCRResult([], "비용은 18,000원입니다.", 0.8)
        with patch.object(processor, "_openai", return_value="비용은 81,000원입니다."):
            self.assertEqual(processor.correct(result), result.text)


if __name__ == "__main__":
    unittest.main()
