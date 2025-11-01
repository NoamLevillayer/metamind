import json
from typing import Any, Dict, List

from analysis.sentiment import metamind_sentiment_json
from baselines.raw_sentiment import baseline_sentiment_json
from config import LLM_CONFIG
from llm_interface import OpenAILLM


def ensure_api_key(config: Dict[str, Any]) -> None:
    api_key = config.get("api_key")
    if not api_key or api_key in {"your_api_key_here", "replace_me"}:
        raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY or update config.py.")


def main() -> None:
    ensure_api_key(LLM_CONFIG)
    llm = OpenAILLM(LLM_CONFIG)

    print("Sentiment demo ready. Press Enter on an empty line to exit.")
    while True:
        review = input("\nUser review: ").strip()
        if not review:
            print("Exiting sentiment demo.")
            break

        context: List[Dict[str, str]] = []
        hypotheses: List[Dict[str, Any]] = []

        baseline_result = baseline_sentiment_json(llm, review, context, max_retries=1)
        metamind_result = metamind_sentiment_json(llm, review, context, hypotheses)

        print("\nBaseline output:")
        print(json.dumps(baseline_result, indent=2))

        print("\nMetaMind output:")
        print(json.dumps(metamind_result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
