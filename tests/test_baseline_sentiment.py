import unittest
from typing import Any, Dict

from baselines.raw_sentiment import baseline_sentiment_json
from llm_interface.base_llm import BaseLLM


class MockLLM(BaseLLM):
    def __init__(self, responses):
        super().__init__({})
        self._responses = list(responses)

    def generate(self, prompt: str, **kwargs) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "{}"  # default empty JSON


class TestBaselineSentiment(unittest.TestCase):
    def test_retry_then_success(self):
        bad = "not json"
        good = '{"polarity": "negative", "intensity": 0.7, "aspects": [{"name": "pricing", "sentiment": "negative", "evidence": "too expensive"}], "evidence": "-", "confidence": 0.8, "source": "baseline"}'
        llm = MockLLM([bad, good])
        result: Dict[str, Any] = baseline_sentiment_json(llm, "it is too expensive", [], max_retries=1)
        self.assertEqual(result.get("polarity"), "negative")
        self.assertEqual(result.get("source"), "baseline")
        self.assertIn("aspects", result)

    def test_fallback_on_failure(self):
        llm = MockLLM(["still not json"])
        result = baseline_sentiment_json(llm, "hello", [], max_retries=0)
        self.assertEqual(result.get("polarity"), "neutral")
        self.assertEqual(result.get("source"), "baseline")
        self.assertIn("raw_text", result)


if __name__ == "__main__":
    unittest.main()

