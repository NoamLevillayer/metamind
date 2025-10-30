import unittest

from analysis.sentiment import heuristic_sentiment_from_hypotheses


class TestSentimentHeuristics(unittest.TestCase):
    def test_positive_overrides_negative(self):
        hypotheses = [
            {"type": "Emotion", "explanation": "The user is happy and says it is great", "score": 0.9},
            {"type": "Belief", "explanation": "It is broken and slow sometimes", "score": 0.4},
        ]
        polarity, conf = heuristic_sentiment_from_hypotheses(hypotheses)
        self.assertIn(polarity, {"positive", "neutral", "negative"})
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        # Expect leaning positive due to strong positive cues and higher weight
        self.assertEqual(polarity, "positive")

    def test_negative_with_desire_and_cues(self):
        hypotheses = [
            {"type": "Desire", "explanation": "I want to cancel and return due to issues", "p_cond": 0.8},
            {"type": "Belief", "explanation": "The app is buggy and confusing", "p_cond": 0.6},
        ]
        polarity, conf = heuristic_sentiment_from_hypotheses(hypotheses)
        self.assertEqual(polarity, "negative")
        self.assertGreater(conf, 0.4)  # should not be the bare minimum


if __name__ == "__main__":
    unittest.main()
