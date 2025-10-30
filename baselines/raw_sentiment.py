from typing import Any, Dict, List
from llm_interface.base_llm import BaseLLM
from prompts.prompt_templates import EXTRA_SENTIMENT_PROMPTS
from utils.helpers import parse_json_from_string

def _format_context(context: List[Dict[str, str]]) -> str:
    if not context:
        return "No previous conversation history."
    lines = []
    for t in context:
        sp = t.get("speaker", "Unknown")
        ut = t.get("utterance", "")
        lines.append(f"{sp}: {ut}")
    return "\n".join(lines)

def baseline_sentiment_json(
    llm: BaseLLM,
    user_input: str,
    conversation_context: List[Dict[str, str]],
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Single-pass baseline sentiment via LLM with strict JSON output and retries.
    - Returns a dict roughly matching SentimentResult schema.
    - Never raises; returns a neutral fallback with raw_text on parse failure.
    """
    C_t = _format_context(conversation_context)
    prompt = EXTRA_SENTIMENT_PROMPTS["BASELINE_SENTIMENT_JSON"].format(u_t=user_input, C_t=C_t)

    last_text = None
    for attempt in range(max_retries + 1):
        last_text = llm.generate(prompt, temperature=0.2, max_tokens=300)
        parsed = parse_json_from_string(last_text)
        if parsed:
            parsed["source"] = "baseline"
            return parsed
        prompt = prompt + "\nIMPORTANT: Return ONLY valid JSON. No extra text."

    return {
        "polarity": "neutral",
        "intensity": 0.5,
        "aspects": [],
        "evidence": "Parsing failed; baseline fallback.",
        "confidence": 0.4,
        "source": "baseline",
        "raw_text": last_text,
    }
