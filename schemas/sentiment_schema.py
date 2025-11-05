from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

VALID_POLARITY = {"positive", "neutral", "negative"}

@dataclass
class AspectSentiment:
    name: str
    sentiment: str  # positive|neutral|negative
    evidence: Optional[str] = None

@dataclass
class SentimentResult:
    polarity: str                     # positive|neutral|negative
    intensity: float                  # 0.0-1.0
    aspects: List[AspectSentiment] = field(default_factory=list)
    evidence: Optional[str] = None
    confidence: float = 0.0           # 0.0-1.0
    source: str = "baseline"          # baseline|metamind

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polarity": self.polarity,
            "intensity": float(max(0.0, min(1.0, self.intensity))),
            "aspects": [
                {"name": a.name, "sentiment": a.sentiment, "evidence": a.evidence}
                for a in self.aspects
            ],
            "evidence": self.evidence,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "source": self.source,
        }

    @staticmethod
    def _norm_polarity(p: Any) -> str:
        s = str(p).strip().lower()
        return s if s in VALID_POLARITY else "neutral"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentimentResult":
        aspects_in = d.get("aspects") or []
        aspects = []
        for a in aspects_in:
            name = str(a.get("name", "")).strip()[:128]
            sentiment = cls._norm_polarity(a.get("sentiment", "neutral"))
            evidence = str(a.get("evidence")) if a.get("evidence") is not None else None
            if name:
                aspects.append(AspectSentiment(name=name, sentiment=sentiment, evidence=evidence))

        return cls(
            polarity=cls._norm_polarity(d.get("polarity", "neutral")),
            intensity=float(d.get("intensity", 0.5)),
            aspects=aspects,
            evidence=str(d.get("evidence")) if d.get("evidence") is not None else None,
            confidence=float(d.get("confidence", 0.5)),
            source=str(d.get("source", "baseline")),
        )
