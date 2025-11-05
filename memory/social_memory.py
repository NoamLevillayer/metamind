import json
import logging
from typing import Any, Dict, List, Optional

from llm_interface.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class SocialMemory:

    def __init__(self, llm_interface: BaseLLM, config: Optional[Dict[str, Any]] = None):
        self.llm = llm_interface
        self.config = config or {}
        self.user_memory: Dict[str, Dict[str, Any]] = {}
        self.summary_prompt = (
            "Based on the stored user preferences, recent emotional markers, and "
            "interaction history, generate a concise summary of the user's social "
            "memory. This summary will be used to inform an AI agent's responses. "
            "Highlight key aspects relevant for personalization and empathetic "
            "interaction.\n\n"
            "Preferences: {preferences}\n"
            "Recent Emotions: {emotions}\n"
            "Interaction History: {history}\n\n"
            "Concise Summary for Agent:"
        )

    def _ensure_user_memory_exists(self, user_id: str) -> None:
        if user_id not in self.user_memory:
            self.user_memory[user_id] = {
                "preferences": {},
                "emotional_markers": [],
                "interaction_history": [],
            }

    def get_recent_emotional_markers(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        self._ensure_user_memory_exists(user_id)
        markers = self.user_memory[user_id]["emotional_markers"]
        sorted_markers = sorted(markers, key=lambda item: item.get("timestamp", ""), reverse=True)
        return sorted_markers[:limit]

    def get_interaction_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._ensure_user_memory_exists(user_id)
        interactions = self.user_memory[user_id]["interaction_history"]
        sorted_interactions = sorted(interactions, key=lambda item: item.get("timestamp", ""), reverse=True)
        return sorted_interactions[:limit]

    def get_summary(
        self,
        user_id: str,
        max_preferences: int = 5,
        max_emotions: int = 3,
        max_interactions: int = 3,
    ) -> str:
        """
        Generate a concise social-memory summary for prompt injection.

        Falls back to a simple stitched string if the LLM call fails.
        """
        self._ensure_user_memory_exists(user_id)

        user_data = self.user_memory[user_id]
        preferences_summary = dict(list(user_data["preferences"].items())[:max_preferences])
        emotions_summary = self.get_recent_emotional_markers(user_id, limit=max_emotions)
        interactions_summary = self.get_interaction_history(user_id, limit=max_interactions)

        prompt = self.summary_prompt.format(
            preferences=json.dumps(preferences_summary, indent=2),
            emotions=json.dumps(emotions_summary, indent=2),
            history=json.dumps(interactions_summary, indent=2),
        )

        try:
            summary = self.llm.generate(prompt, max_tokens=200)
            return summary.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "[SocialMemory] LLM failed to generate prompt summary for user %s: %s",
                user_id,
                exc,
            )
            return (
                f"User Preferences: {json.dumps(preferences_summary, indent=2)}\n"
                f"Recent Emotions: {json.dumps(emotions_summary, indent=2)}\n"
                f"Recent Interactions: {json.dumps(interactions_summary, indent=2)}"
            )
