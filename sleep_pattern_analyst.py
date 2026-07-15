"""
sleep_pattern_analyst.py
--------------------------
AI Sleep Cycle Optimizer & Data Analyst
(Option 3: "Sleep Pattern Detection" ka core module)

Flow:
1. SQLite (sleep_ai.db) se pichle 30 din ka sleep data load karo (pandas)
2. Trend analysis - last 7 din vs pichle 7 din
3. Correlation analysis - caffeine time vs sleep efficiency, workout vs deep sleep
4. Rule-based threshold flags
5. Sab kuch Gemini ko bhejo -> ek friendly, personalized Hindi-English tip milta hai
"""

import os
import sqlite3
import json
import pandas as pd
import requests
from typing import Optional

MODEL_NAME = "gemini-2.5-flash"
DB_PATH = "sleep_ai.db"


def load_recent_data(user_id: str = "anonymous", days: int = 30, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load historical sessions from the SQLite database."""
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df_raw = pd.read_sql_query(
        "SELECT timestamp, sleep_metrics, mood_metrics FROM sessions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        conn, params=(user_id, days)
    )
    conn.close()

    if df_raw.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df_raw.iterrows():
        try:
            sm = json.loads(row["sleep_metrics"] or "{}")
        except Exception:
            sm = {}
        try:
            mm = json.loads(row["mood_metrics"] or "{}")
        except Exception:
            mm = {}

        # Skip empty metrics rows to prevent division errors
        if not sm and not mm:
            continue

        rows.append({
            "date":                 row["timestamp"][:10],
            "total_sleep_hours":    sm.get("duration", 8.0),
            "rem_percent":          sm.get("rem_percent", 20.0),
            "deep_sleep_percent":   sm.get("deep_percent", 15.0),
            "light_sleep_percent":  sm.get("light_percent", 65.0),
            "avg_heart_rate":       sm.get("hr", mm.get("hr", 65.0)),
            "sleep_efficiency":     sm.get("efficiency", 85.0),
            "caffeine_hour":        sm.get("caffeine_hour", None),
            "workout_evening":      sm.get("workout_evening", None),
            "predicted_mood":       mm.get("mood", "Neutral")
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_trends(df: pd.DataFrame) -> dict:
    if len(df) < 4:  # Minimum 4 records to compute some basic trends
        return {"note": "Not enough data yet for a trend comparison."}

    # Split into recent half and older half
    half = len(df) // 2
    recent_half = df.tail(half)
    prev_half = df.head(len(df) - half)

    trends = {}
    for col in ["rem_percent", "deep_sleep_percent", "sleep_efficiency", "total_sleep_hours"]:
        recent_avg = recent_half[col].mean()
        prev_avg = prev_half[col].mean()
        trends[col] = {
            "recent_avg": round(float(recent_avg), 2),
            "previous_avg": round(float(prev_avg), 2),
            "change": round(float(recent_avg - prev_avg), 2),
        }
    return trends


def compute_correlations(df: pd.DataFrame) -> dict:
    correlations = {}
    
    # We need at least 3 non-null records for correlations
    valid_caff = df[df["caffeine_hour"].notna()]
    if len(valid_caff) >= 3:
        try:
            correlations["caffeine_vs_efficiency"] = round(
                float(valid_caff["caffeine_hour"].astype(float).corr(valid_caff["sleep_efficiency"].astype(float))), 2
            )
        except Exception:
            pass

    valid_workout = df[df["workout_evening"].notna()]
    if len(valid_workout) >= 3:
        try:
            correlations["workout_vs_deep_sleep"] = round(
                float(valid_workout["workout_evening"].astype(float).corr(valid_workout["deep_sleep_percent"].astype(float))), 2
            )
        except Exception:
            pass

    # Clean nan values
    cleaned = {}
    for k, v in correlations.items():
        import math
        if not math.isnan(v):
            cleaned[k] = v
    return cleaned


def generate_flags(df: pd.DataFrame, correlations: dict) -> list:
    flags = []
    if df.empty:
        return flags

    latest = df.iloc[-1]

    if latest["rem_percent"] < 15:
        flags.append("REM sleep is below the healthy 15% threshold.")
    if latest["deep_sleep_percent"] < 10:
        flags.append("Deep sleep is below the healthy 10-13% range.")
    if latest["sleep_efficiency"] < 80:
        flags.append("Sleep efficiency is below 80%, indicating restless sleep.")

    caff_corr = correlations.get("caffeine_vs_efficiency")
    if caff_corr is not None and caff_corr < -0.2:
        flags.append("Later caffeine intake is correlated with lower sleep efficiency.")

    workout_corr = correlations.get("workout_vs_deep_sleep")
    if workout_corr is not None and workout_corr < -0.2:
        flags.append("Evening workouts appear correlated with reduced deep sleep.")

    return flags


def generate_personalized_report(trends: dict, correlations: dict, flags: list, api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return (
            "Please configure your Gemini API key in the sidebar to receive "
            "AI-powered custom trends and daily circadian cutoff times."
        )

    prompt = f"""
You are a friendly sleep-health assistant inside a sleep-tracking dashboard app.

Here is the user's recent sleep analysis:

Trends (recent avg vs previous avg):
{json.dumps(trends, indent=2)}

Correlations detected:
{json.dumps(correlations, indent=2)}

Rule-based flags triggered:
{json.dumps(flags, indent=2)}

Write a short (3-5 sentence), warm, encouraging daily sleep tip in a natural
Hindi-English mixed tone (Hinglish), the way a caring friend would explain it.
Include ONE specific, actionable suggestion (like a caffeine cutoff time or
an evening activity change) based on the data above. Do not use bullet points,
write it as flowing conversational text. Do not repeat raw numbers robotically -
weave them in naturally.
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return f"Unable to generate personalized report (API status {response.status_code})."
    except Exception as exc:
        return f"Error analyzing trends: {str(exc)}"


def run_full_analysis(user_id: str = "anonymous", api_key: Optional[str] = None, db_path: str = DB_PATH) -> dict:
    df = load_recent_data(user_id=user_id, days=30, db_path=db_path)

    if df.empty or len(df) < 2:
        return {
            "error": "Not enough sleep data found yet. Log at least 2 sessions to run trend analysis.",
            "trends": {},
            "correlations": {},
            "flags": [],
            "personalized_report": "Log more nights of sleep to unlock custom trends and circadian cutoff recommendations!"
        }

    trends = compute_trends(df)
    correlations = compute_correlations(df)
    flags = generate_flags(df, correlations)
    report = generate_personalized_report(trends, correlations, flags, api_key=api_key)

    return {
        "trends": trends,
        "correlations": correlations,
        "flags": flags,
        "personalized_report": report,
    }
