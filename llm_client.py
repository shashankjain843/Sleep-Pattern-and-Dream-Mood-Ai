"""
llm_client.py — Optional Anthropic Claude integration.

All functions return None / empty on failure, so callers always
fall back to rule-based behaviour when no API key is configured.

Set env var:  ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_MODEL = "claude-3-5-haiku-20241022"   # fast + cheap model


def is_available() -> bool:
    """Return True if the Anthropic API key is set and the library is installed."""
    if not _ANTHROPIC_API_KEY:
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _get_client():
    try:
        import anthropic
        return anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)
    except Exception as exc:
        logger.warning("Could not create Anthropic client: %s", exc)
        return None


# ── Feature 1: Dynamic narrative report paragraph ────────────────────────────
def generate_narrative(sleep_metrics: dict, mood_metrics: dict) -> Optional[str]:
    """
    Return a natural, personalised narrative paragraph summarising the session.
    Falls back to None if unavailable.
    """
    if not is_available():
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        prompt = (
            "You are a sleep-health assistant. Write ONE concise paragraph (3-5 sentences) "
            "that gives a personalised, warm, and actionable summary of this person's session.\n\n"
            f"Sleep metrics: {sleep_metrics}\n"
            f"Mood metrics: {mood_metrics}\n\n"
            "Stick only to these numbers — do not invent data. Be direct and helpful."
        )
        response = client.messages.create(
            model=_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Claude narrative generation failed: %s", exc)
        return None


# ── Feature 2: Q&A about the user's own results ──────────────────────────────
def answer_insight(question: str, session_metrics: dict) -> Optional[str]:
    """
    Answer a user question grounded only in their session data.
    Returns None on failure.
    """
    if not is_available():
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        system = (
            "You are a sleep-health assistant. Answer the user's question using ONLY "
            "the session data provided. Do not add medical advice beyond what the data supports. "
            "Be concise (2-4 sentences)."
        )
        user_msg = (
            f"Session data:\n{session_metrics}\n\n"
            f"User question: {question}"
        )
        response = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Claude Q&A failed: %s", exc)
        return None


# ── Feature 3: Cross-metric correlated suggestions ───────────────────────────
def generate_correlated_suggestions(full_features: dict) -> Optional[list[str]]:
    """
    Generate suggestions that reflect cross-metric patterns
    (e.g. high stress + low deep sleep together).
    Returns a list of suggestion strings, or None on failure.
    """
    if not is_available():
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        prompt = (
            "You are a sleep-health assistant. Analyse these physiological features "
            "and return exactly 3 actionable suggestions that reflect CROSS-METRIC patterns "
            "(e.g. combinations of metrics, not just individual thresholds). "
            "Return them as a JSON array of strings — nothing else.\n\n"
            f"Features: {full_features}"
        )
        response = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        suggestions = json.loads(text)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions]
        return None
    except Exception as exc:
        logger.warning("Claude correlated suggestions failed: %s", exc)
        return None
