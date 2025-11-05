"""Helpers to turn MetaMind sentiment recommendations into plain text."""

from __future__ import annotations

from typing import Any, Dict, List


def _format_recommendation_text(recommendation: Dict[str, Any]) -> str:
    """Compose a compact human-readable string from a recommendation payload."""
    if not recommendation:
        return "No recommendation available."

    actions = recommendation.get("actions") or []
    action_lines = [str(action).strip() for action in actions if str(action).strip()]

    if action_lines:
        first_action = action_lines[0]
        return f"Consider this: {first_action}"

    return "No recommendation available."


def recommendation_text_from_result(metamind_result: Dict[str, Any]) -> str:
    """Convert the recommendation portion of a MetaMind result into plain text."""
    recommendation = metamind_result.get("recommendation", {}) if metamind_result else {}
    return _format_recommendation_text(recommendation)


__all__ = ["recommendation_text_from_result"]
