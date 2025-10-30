from typing import Any, Dict, List, Tuple, Optional
from llm_interface.base_llm import BaseLLM
from prompts.prompt_templates import EXTRA_SENTIMENT_PROMPTS
from utils.helpers import parse_json_from_string
from config import SENTIMENT_CUES, SENTIMENT_WEIGHTS

def _normalize_weights(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [v / total for v in values]


def heuristic_sentiment_from_hypotheses(hypotheses: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Deterministic fallback sentiment from ToM/Domain hypotheses using config-driven
    cues and weights. Returns (polarity, confidence).
    """
    if not hypotheses:
        return "neutral", 0.4

    # Normalize weights using score if available, else p_cond
    weights: List[float] = []
    for h in hypotheses:
        try:
            w = float(h.get("score", h.get("p_cond", 0.5)))
        except Exception:
            w = 0.5
        weights.append(max(0.0, w))
    norm_weights = _normalize_weights(weights)

    # Cues from config
    base_pos = [c.lower() for c in SENTIMENT_CUES.get("positive", [])]
    base_neg = [c.lower() for c in SENTIMENT_CUES.get("negative", [])]
    emo_pos = [c.lower() for c in SENTIMENT_CUES.get("emotion", {}).get("positive", [])]
    emo_neg = [c.lower() for c in SENTIMENT_CUES.get("emotion", {}).get("negative", [])]
    int_pos = [c.lower() for c in SENTIMENT_CUES.get("intention", {}).get("positive", [])]
    int_neg = [c.lower() for c in SENTIMENT_CUES.get("intention", {}).get("negative", [])]
    bel_pos = [c.lower() for c in SENTIMENT_CUES.get("belief", {}).get("positive", [])]
    bel_neg = [c.lower() for c in SENTIMENT_CUES.get("belief", {}).get("negative", [])]

    # Weights from config
    w_base_pos = float(SENTIMENT_WEIGHTS.get("base_positive_cue", 0.5))
    w_base_neg = float(SENTIMENT_WEIGHTS.get("base_negative_cue", 0.5))
    w_emo_pos = float(SENTIMENT_WEIGHTS.get("emotion_positive", 0.4))
    w_emo_neg = float(SENTIMENT_WEIGHTS.get("emotion_negative", 0.4))
    w_int_pos = float(SENTIMENT_WEIGHTS.get("intention_positive", 0.3))
    w_int_neg = float(SENTIMENT_WEIGHTS.get("intention_negative", 0.3))
    w_bel_pos = float(SENTIMENT_WEIGHTS.get("belief_positive", 0.3))
    w_bel_neg = float(SENTIMENT_WEIGHTS.get("belief_negative", 0.3))
    w_desire_prior_neg = float(SENTIMENT_WEIGHTS.get("desire_neg_prior", 0.2))

    pos = 0.0
    neg = 0.0

    for h, w in zip(hypotheses, norm_weights):
        t = str(h.get("type", "Unknown")).lower()
        expl = str(h.get("explanation", "")).lower()

        if t == "desire":
            neg += w_desire_prior_neg * w
        if t == "emotion":
            if any(k in expl for k in emo_pos):
                pos += w_emo_pos * w
            if any(k in expl for k in emo_neg):
                neg += w_emo_neg * w
        elif t == "intention":
            if any(k in expl for k in int_pos):
                pos += w_int_pos * w
            if any(k in expl for k in int_neg):
                neg += w_int_neg * w
        elif t == "belief":
            if any(k in expl for k in bel_pos):
                pos += w_bel_pos * w
            if any(k in expl for k in bel_neg):
                neg += w_bel_neg * w

        if any(k in expl for k in base_pos):
            pos += w_base_pos * w
        if any(k in expl for k in base_neg):
            neg += w_base_neg * w

    gap = pos - neg
    mag = max(pos, neg)
    if abs(gap) < 0.1:
        return "neutral", min(1.0, 0.5 + 0.3 * mag)
    if gap > 0:
        return "positive", min(1.0, 0.6 + 0.3 * pos)
    return "negative", min(1.0, 0.6 + 0.3 * neg)

def _format_context(context: List[Dict[str, str]]) -> str:
    if not context:
        return "No previous conversation history."
    return "\n".join(f"{t.get('speaker','Unknown')}: {t.get('utterance','')}" for t in context)

def _top_k_hypotheses(hypotheses: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    if not hypotheses:
        return []
    def key(h):
        return float(h.get("score", h.get("p_cond", 0.0)))
    return sorted(hypotheses, key=key, reverse=True)[:k]

def synthesize_with_llm(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    hypotheses: List[Dict[str, Any]],
    max_retries: int = 1
) -> Optional[Dict[str, Any]]:
    C_t = _format_context(conversation_context)
    top = _top_k_hypotheses(hypotheses)
    prompt = EXTRA_SENTIMENT_PROMPTS["SENTIMENT_SYNTHESIS_JSON"].format(
        u_t=user_input,
        C_t=C_t,
    ) + "\nTop Hypotheses (H):\n" + str(top)

    last_text = None
    for _ in range(max_retries + 1):
        last_text = llm.generate(prompt, temperature=0.3, max_tokens=350)
        parsed = parse_json_from_string(last_text)
        if parsed:
            parsed["source"] = "metamind"
            return parsed
        prompt = prompt + "\nReminder: Return ONLY valid JSON."

    return None

def extract_aspects_with_llm(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    max_retries: int = 1
) -> List[Dict[str, Any]]:
    C_t = _format_context(conversation_context)
    prompt = EXTRA_SENTIMENT_PROMPTS["ASPECT_EXTRACTION_JSON"].format(u_t=user_input, C_t=C_t)

    last_text = None
    for _ in range(max_retries + 1):
        last_text = llm.generate(prompt, temperature=0.3, max_tokens=250)
        parsed = parse_json_from_string(last_text)
        if parsed and isinstance(parsed.get("aspects"), list):
            return parsed["aspects"]
        prompt = prompt + "\nIMPORTANT: Output ONLY JSON matching the schema."

    return []

def metamind_sentiment_json(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    hypotheses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    End-to-end MetaMind sentiment using both LLM synthesis and heuristic fallback.

    Returns an approximate SentimentResult-like dict. Pipeline may normalize via schema.
    """
    # Heuristic first
    pol, conf0 = heuristic_sentiment_from_hypotheses(hypotheses)
    # Try LLM synthesis
    synth = synthesize_with_llm(llm, user_input, conversation_context, hypotheses, max_retries=1)
    # Aspects
    aspects = extract_aspects_with_llm(llm, user_input, conversation_context, max_retries=1)

    if synth:
        # Merge aspects if missing/empty
        if not synth.get("aspects"):
            synth["aspects"] = aspects
        synth.setdefault("polarity", pol)
        synth.setdefault("intensity", 0.6 if pol != "neutral" else 0.5)
        synth.setdefault("evidence", "Consolidated from top hypotheses.")
        synth["confidence"] = max(float(synth.get("confidence", 0.6)), conf0)
        synth["source"] = "metamind"
        return synth

    # Fallback: heuristic + aspects
    return {
        "polarity": pol,
        "intensity": 0.6 if pol != "neutral" else 0.5,
        "aspects": aspects,
        "evidence": "Heuristic fallback from hypotheses.",
        "confidence": conf0,
        "source": "metamind",
    }
