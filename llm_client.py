"""
llm_client.py — Optional Anthropic Claude and Google Gemini integration.

All functions return None / empty on failure, so callers always
fall back to rule-based behaviour when no API key is configured.

Set env vars:
  - ANTHROPIC_API_KEY=sk-ant-...
  - GEMINI_API_KEY=AIzaSy...
"""
import os
import logging
import requests
import json
from typing import Optional

logger = logging.getLogger(__name__)

_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_MODEL = "claude-3-5-haiku-20241022"   # fast + cheap model


def is_available(api_key: Optional[str] = None, provider: str = "claude") -> bool:
    """Return True if the API key is set/provided and the provider is supported."""
    key = api_key
    if not key:
        key = _GEMINI_API_KEY if provider == "gemini" else _ANTHROPIC_API_KEY

    if not key:
        return False

    if provider == "claude":
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False
    elif provider == "gemini":
        return True
    return False


def _get_client(api_key: Optional[str] = None):
    key = api_key or _ANTHROPIC_API_KEY
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except Exception as exc:
        logger.warning("Could not create Anthropic client: %s", exc)
        return None


def _call_gemini_api(prompt: str, api_key: str, system_instruction: Optional[str] = None) -> Optional[str]:
    """Helper to call Gemini API directly using HTTP requests."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    contents = {
        "parts": [{"text": prompt}]
    }
    
    payload = {
        "contents": [contents]
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
        
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates")
            if candidates and len(candidates) > 0:
                content = candidates[0].get("content")
                if content:
                    parts = content.get("parts")
                    if parts and len(parts) > 0:
                        text = parts[0].get("text")
                        return text.strip()
            logger.warning("Gemini response missing text: %s", data)
            return None
        else:
            logger.warning("Gemini API failed with status %d: %s", response.status_code, response.text)
            return None
    except Exception as exc:
        logger.warning("Gemini API call exception: %s", exc)
        return None


# ── Feature 1: Dynamic narrative report paragraph ────────────────────────────
def generate_narrative(
    sleep_metrics: dict, 
    mood_metrics: dict, 
    api_key: Optional[str] = None, 
    provider: str = "claude"
) -> Optional[str]:
    """
    Return a natural, personalised narrative paragraph summarising the session.
    Falls back to None if unavailable.
    """
    effective_key = api_key
    effective_provider = provider
    if not effective_key:
        if _GEMINI_API_KEY:
            effective_key = _GEMINI_API_KEY
            effective_provider = "gemini"
        elif _ANTHROPIC_API_KEY:
            effective_key = _ANTHROPIC_API_KEY
            effective_provider = "claude"

    if not is_available(effective_key, effective_provider):
        return None

    prompt = (
        "You are a sleep-health assistant. Write ONE concise paragraph (3-5 sentences) "
        "that gives a personalised, warm, and actionable summary of this person's session.\n\n"
        f"Sleep metrics: {sleep_metrics}\n"
        f"Mood metrics: {mood_metrics}\n\n"
        "Stick only to these numbers — do not invent data. Be direct and helpful."
    )

    if effective_provider == "gemini":
        return _call_gemini_api(prompt, effective_key)
    
    # Claude fallback
    client = _get_client(effective_key)
    if client is None:
        return None
    try:
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
def answer_insight(
    question: str, 
    session_metrics: dict, 
    api_key: Optional[str] = None, 
    provider: str = "claude"
) -> Optional[str]:
    """
    Answer a user question grounded only in their session data.
    Returns None on failure.
    """
    effective_key = api_key
    effective_provider = provider
    if not effective_key:
        if _GEMINI_API_KEY:
            effective_key = _GEMINI_API_KEY
            effective_provider = "gemini"
        elif _ANTHROPIC_API_KEY:
            effective_key = _ANTHROPIC_API_KEY
            effective_provider = "claude"

    if not is_available(effective_key, effective_provider):
        return None

    system = (
        "You are a sleep-health assistant. Answer the user's question using ONLY "
        "the session data provided. Do not add medical advice beyond what the data supports. "
        "Be concise (2-4 sentences)."
    )
    user_msg = (
        f"Session data:\n{session_metrics}\n\n"
        f"User question: {question}"
    )

    if effective_provider == "gemini":
        # Combine system and user msg for Gemini since we pass systemInstruction separately
        return _call_gemini_api(user_msg, effective_key, system_instruction=system)

    # Claude fallback
    client = _get_client(effective_key)
    if client is None:
        return None
    try:
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
def generate_correlated_suggestions(
    full_features: dict, 
    api_key: Optional[str] = None, 
    provider: str = "claude"
) -> Optional[list[str]]:
    """
    Generate suggestions that reflect cross-metric patterns
    (e.g. high stress + low deep sleep together).
    Returns a list of suggestion strings, or None on failure.
    """
    effective_key = api_key
    effective_provider = provider
    if not effective_key:
        if _GEMINI_API_KEY:
            effective_key = _GEMINI_API_KEY
            effective_provider = "gemini"
        elif _ANTHROPIC_API_KEY:
            effective_key = _ANTHROPIC_API_KEY
            effective_provider = "claude"

    if not is_available(effective_key, effective_provider):
        return None

    prompt = (
        "You are a sleep-health assistant. Analyse these physiological features "
        "and return exactly 3 actionable suggestions that reflect CROSS-METRIC patterns "
        "(e.g. combinations of metrics, not just individual thresholds). "
        "Return them as a JSON array of strings — nothing else.\n\n"
        f"Features: {full_features}"
    )

    text = None
    if effective_provider == "gemini":
        # Request JSON output structure
        text = _call_gemini_api(prompt + "\nReturn ONLY raw JSON. No codeblocks.", effective_key)
    else:
        client = _get_client(effective_key)
        if client is not None:
            try:
                response = client.messages.create(
                    model=_MODEL,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
            except Exception as exc:
                logger.warning("Claude correlated suggestions failed: %s", exc)

    if not text:
        return None

    try:
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
        logger.warning("JSON parsing of suggestions failed: %s | Response: %s", exc, text)
        return None
