"""
dream_mood_module.py
---------------------
Visual Dream Journal & AI Art Generator
(Option 1: "Dream Mood Prediction" ka core module)

Flow:
1. User apne sapne ke baare mein 1-2 line likhta hai
2. Sleep data (REM %, Heart Rate) + predicted mood us text ke saath merge hota hai
3. Gemini API dream ka psychological interpretation deti hai
4. Gemini hi ek short image-generation prompt bhi banati hai
5. Pollinations.ai (free, no API key needed) us prompt se live image generate karta hai
"""

import os
import json
import urllib.parse
import requests
from typing import Optional

MODEL_NAME = "gemini-2.5-flash"  # fast + cheap, good enough for this task
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt/"


def analyze_dream(dream_text: str, mood: str, rem_percent: float, heart_rate: int, api_key: Optional[str] = None) -> dict:
    """
    Dream text + sleep data + predicted mood ko Gemini ko bhejta hai.
    Uses direct requests to avoid external google-generativeai library dependencies on Python 3.14.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {
            "interpretation": "Please enter a Gemini API Key in the sidebar API Settings to analyze your dream.",
            "image_prompt": "fantasy surreal painting, digital art, dreamlike",
            "mood_tag": mood or "Unknown",
        }

    system_prompt = f"""
You are a thoughtful dream-interpretation assistant used inside a sleep-tracking app.

User's biometric context:
- Predicted mood after waking: {mood}
- REM sleep percentage last night: {rem_percent}%
- Average heart rate during sleep: {heart_rate} bpm

User's dream description (their own words): "{dream_text}"

Do THREE things and return ONLY valid JSON (no markdown fences, no extra text):
{{
  "interpretation": "2-4 sentence gentle psychological interpretation of the dream,
                      written in a warm Hindi-English mixed tone, connecting it to
                      the sleep quality and mood context. Do not sound clinical.",
  "image_prompt": "A vivid, purely visual English prompt (max 40 words) describing
                    the dream as an artistic scene, suitable for an AI image generator.
                    Mention style like 'digital art, dreamlike, soft lighting'.",
  "mood_tag": "one or two word emotional tag capturing the dream's feel, e.g.
               'Wonder', 'Anxious Flight', 'Peaceful Floating'"
}}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": system_prompt}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            raw_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
            # Clean json fences if present
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw_text)
            return result
        else:
            return {
                "interpretation": f"Could not interpret dream (API status {response.status_code}).",
                "image_prompt": f"{dream_text}, digital art, dreamlike, soft lighting",
                "mood_tag": mood,
            }
    except Exception as exc:
        return {
            "interpretation": f"Error during interpretation: {str(exc)}",
            "image_prompt": f"{dream_text}, digital art, dreamlike, soft lighting",
            "mood_tag": mood,
        }


def generate_dream_image_url(image_prompt: str, width: int = 768, height: int = 512, seed: Optional[int] = None) -> str:
    """
    Pollinations.ai ek simple GET request se image deta hai.
    """
    encoded_prompt = urllib.parse.quote(image_prompt)
    url = f"{POLLINATIONS_BASE_URL}{encoded_prompt}?width={width}&height={height}&nologo=true"
    if seed is not None:
        url += f"&seed={seed}"
    return url


def process_dream_entry(dream_text: str, mood: str, rem_percent: float, heart_rate: int, api_key: Optional[str] = None) -> dict:
    """
    Poora pipeline: text -> Gemini analysis -> image prompt -> Pollinations image URL
    """
    analysis = analyze_dream(dream_text, mood, rem_percent, heart_rate, api_key=api_key)
    image_url = generate_dream_image_url(analysis["image_prompt"])

    return {
        "dream_text": dream_text,
        "mood": mood,
        "rem_percent": rem_percent,
        "heart_rate": heart_rate,
        "interpretation": analysis["interpretation"],
        "mood_tag": analysis["mood_tag"],
        "image_prompt": analysis["image_prompt"],
        "image_url": image_url,
    }
