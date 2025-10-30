import unittest
from typing import Any, Dict, List

from analysis.sentiment import metamind_sentiment_json
from llm_interface.base_llm import BaseLLM


class QueueLLM(BaseLLM):
    """Returns queued responses per call to generate."""
    def __init__(self, responses: List[str]):
        super().__init__({})
        self._q = list(responses)

    def generate(self, prompt: str, **kwargs) -> str:
        return self._q.pop(0) if self._q else "{}"


class TestSynthesisLLM(unittest.TestCase):
    def test_synthesis_and_aspects_merge(self):
        # First call (synthesis) returns JSON without aspects; second returns aspects JSON
        synth_json = '{"polarity": "positive", "intensity": 0.8, "evidence": "-", "confidence": 0.7, "source": "metamind"}'
        aspects_json = '{"aspects": [{"name": "quality", "sentiment": "positive", "evidence": "solid build"}]}'
        llm = QueueLLM([synth_json, aspects_json])

        hypotheses = [
            {"type": "Emotion", "explanation": "happy with great quality", "score": 0.9},
            {"type": "Belief", "explanation": "it is reliable", "p_cond": 0.6},
        ]
        ctx: List[Dict[str, str]] = [{"speaker": "User", "utterance": "I love the build quality"}]
        result = metamind_sentiment_json(llm, "I love the build quality", ctx, hypotheses)
        self.assertEqual(result.get("source"), "metamind")
        self.assertEqual(result.get("polarity"), "positive")
        self.assertTrue(result.get("aspects"))

    def test_synthesis_failure_fallback_heuristic(self):
        # First call fails (invalid), second returns aspects JSON
        llm = QueueLLM(["not json", '{"aspects": [{"name": "pricing", "sentiment": "negative", "evidence": "too expensive"}]}' ])

        hypotheses = [
            {"type": "Desire", "explanation": "I want to cancel", "score": 0.8},
            {"type": "Belief", "explanation": "it is broken", "p_cond": 0.7},
        ]
        res = metamind_sentiment_json(llm, "I want to cancel, it's broken", [], hypotheses)
        self.assertEqual(res.get("source"), "metamind")
        self.assertEqual(res.get("polarity"), "negative")
        self.assertTrue(res.get("aspects"))


if __name__ == "__main__":
    unittest.main()
