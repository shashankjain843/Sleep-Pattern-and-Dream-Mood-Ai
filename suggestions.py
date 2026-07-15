# File: suggestions.py
import random
import logging
from typing import Optional
import llm_client

logger = logging.getLogger(__name__)


def generate_sleep_suggestions(metrics: dict) -> list[str]:
    """
    Rule-based sleep suggestions.
    Input keys: rem_percent, deep_percent, sleep_efficiency (or efficiency),
                total_sleep_time (or duration), awakenings.
    """
    suggestions = []

    rem        = metrics.get("rem_percent",     metrics.get("rem", 0))
    deep       = metrics.get("deep_percent",    metrics.get("deep", 0))
    eff        = metrics.get("sleep_efficiency", metrics.get("efficiency", 0))
    total_time = metrics.get("total_sleep_time", metrics.get("duration", 0))
    awakenings = metrics.get("awakenings", 0)

    if rem < 20:
        suggestions.append(
            "🧠 **Boost Dream Sleep**: Your REM sleep is low. "
            "Try to keep a consistent sleep schedule to improve memory consolidation."
        )
    elif rem > 30:
        suggestions.append(
            "🧠 **High REM Activity**: High REM can indicate sleep deprivation rebound. "
            "Ensure you're getting enough total sleep."
        )

    if deep < 15:
        suggestions.append(
            "💪 **Deep Rest Needed**: Deep sleep is vital for physical recovery. "
            "Keep your bedroom cool and dark."
        )

    if eff < 85:
        suggestions.append(
            "📉 **Improve Quality**: Sleep efficiency is low. "
            "Limit screen time and avoid caffeine 6 hours before bed."
        )

    if total_time < 6:
        suggestions.append(
            "⏰ **Sleep More**: You're getting less than 6 hours. "
            "Aim for 7–9 hours for optimal health."
        )

    if awakenings > 2:
        suggestions.append(
            "🌙 **Fragmented Sleep**: Frequent awakenings detected. "
            "Check for noise or light disturbances in your room."
        )

    if not suggestions:
        suggestions.append(
            "✨ **Great Sleep!**: Your metrics are healthy. Keep up your current routine."
        )

    return suggestions


def generate_mood_suggestions(mood_label: str, mood_probs: dict) -> list[str]:
    """Rule-based mood suggestions."""
    suggestions = []

    if mood_label == "Positive":
        suggestions.append("🌟 **Harness Positivity**: Use this energy for creative projects or social connection.")
        suggestions.append("📝 **Gratitude Journal**: Write down 3 good things that happened today.")
    elif mood_label == "Neutral":
        suggestions.append("⚖️ **Stay Balanced**: A short walk or listening to music can help maintain your equilibrium.")
        suggestions.append("🎧 **Mental Engagement**: Try a puzzle or a podcast to stimulate your mind.")
    elif mood_label == "Negative":
        suggestions.append("🧘 **Stress Buster**: Try 4-7-8 breathing: Inhale 4s, hold 7s, exhale 8s.")
        suggestions.append("🌳 **Nature Walk**: Step outside for 10 minutes to naturally lower cortisol levels.")
        suggestions.append("⚠️ **Self-Compassion**: Be kind to yourself today. It's okay to take a break.")

    return suggestions


def generate_full_report(
    sleep_metrics: dict, 
    mood_metrics: dict, 
    gemini_key: Optional[str] = None, 
    anthropic_key: Optional[str] = None
) -> dict:
    """
    Generates the full report dict.
    If GEMINI_API_KEY / ANTHROPIC_API_KEY (or custom keys) are set, attempts to 
    replace the templated narrative with an AI-generated personalised paragraph.
    """
    # Normalise key names
    s_metrics = sleep_metrics.copy()
    if "efficiency" in s_metrics and "sleep_efficiency" not in s_metrics:
        s_metrics["sleep_efficiency"] = s_metrics["efficiency"]
    if "duration" in s_metrics and "total_sleep_time" not in s_metrics:
        s_metrics["total_sleep_time"] = s_metrics["duration"]

    sleep_suggestions = generate_sleep_suggestions(s_metrics)
    mood_label  = mood_metrics.get("mood", "Unknown")
    mood_probs  = mood_metrics.get("probabilities", {})
    mood_suggestions = generate_mood_suggestions(mood_label, mood_probs)

    eff  = s_metrics.get("sleep_efficiency", 0)
    deep = s_metrics.get("deep_percent", 0)

    # ── Templated summaries (always generated as fallback) ───────────────────
    sleep_summary = (
        f"Your sleep quality score is **{eff}%**. "
        f"You achieved **{round(deep, 1)}%** deep sleep, which is crucial for physical restoration."
    )
    mood_summary = (
        f"Your predicted dream mood is **{mood_label}**. "
        "This reflects your physiological state during sleep."
    )

    combined_insights = "Your sleep and mood are interconnected. "
    if eff < 80 and mood_label == "Negative":
        combined_insights += (
            "Low sleep quality may be contributing to higher stress levels. "
            "Prioritise rest to improve your emotional state."
        )
    elif eff > 85 and mood_label == "Positive":
        combined_insights += "Great sleep quality is likely supporting your positive emotional balance."
    else:
        combined_insights += "Maintaining a consistent sleep schedule can help stabilise your mood."

    # Determine provider and key
    api_key = gemini_key or llm_client._GEMINI_API_KEY
    provider = "gemini"
    if not api_key:
        api_key = anthropic_key or llm_client._ANTHROPIC_API_KEY
        provider = "claude"

    # ── Optional AI narrative (replaces template if available) ────────────────
    narrative: Optional[str] = None
    if llm_client.is_available(api_key, provider):
        try:
            narrative = llm_client.generate_narrative(s_metrics, mood_metrics, api_key=api_key, provider=provider)
            if narrative:
                logger.info("%s narrative generated successfully", provider.capitalize())
        except Exception:
            logger.warning("%s narrative failed — using template", provider.capitalize(), exc_info=True)

    # ── Optional AI correlated suggestions ────────────────────────────────────
    all_suggestions = sleep_suggestions + mood_suggestions
    if llm_client.is_available(api_key, provider):
        try:
            full_feats = {**s_metrics, **mood_metrics}
            correlated = llm_client.generate_correlated_suggestions(full_feats, api_key=api_key, provider=provider)
            if correlated:
                all_suggestions = correlated + all_suggestions  # AI suggestions first
                logger.info("%s correlated suggestions added", provider.capitalize())
        except Exception:
            logger.warning("%s correlated suggestions failed — using rule-based", provider.capitalize(), exc_info=True)

    # ── Warnings ──────────────────────────────────────────────────────────────
    warnings = []
    if eff < 75:          warnings.append("⚠️ Low Sleep Efficiency")
    if deep < 10:         warnings.append("⚠️ Insufficient Deep Sleep")
    if mood_label == "Negative": warnings.append("⚠️ High Stress Indicators")

    return {
        "title":            "Daily Health & Wellness Report",
        "summary":          "Here is your personalised analysis based on your bio-metrics.",
        "sleep_summary":    sleep_summary,
        "mood_summary":     mood_summary,
        "combined_insights": combined_insights,
        "narrative":        narrative,           # None if AI unavailable
        "suggestions_list": all_suggestions,
        "warnings":         warnings,
        "claude_powered":   narrative is not None,  # Keep key name for backward compatibility
        "ai_provider":      provider if narrative else None
    }
