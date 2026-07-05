# File: run_inference.py
import logging
import numpy as np
import pandas as pd
import joblib
import json
import os
from extract_features import compute_hrv, compute_gsr
from preprocess_sleepedf import load_and_preprocess_file
from preprocess_yaad import load_single_yaad_trial

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
    logger.warning(
        "TensorFlow not available (Python 3.13+ not supported). "
        "Sleep CNN model will be disabled. Simulation mode works normally."
    )

# Model paths
SLEEP_MODEL_PATH = "models/sleep_cnn_lstm/model.h5"
MOOD_MODEL_PATH  = "models/mood_xgb.model"


# ── Model loading (called once at startup) ────────────────────────────────────
def load_models():
    sleep_model = None
    mood_model  = None

    if TF_AVAILABLE and os.path.exists(SLEEP_MODEL_PATH):
        sleep_model = tf.keras.models.load_model(SLEEP_MODEL_PATH)
        logger.info("Sleep CNN-LSTM model loaded from %s", SLEEP_MODEL_PATH)
    elif not TF_AVAILABLE:
        logger.info("Sleep CNN model skipped (TensorFlow not installed). Simulation mode active.")
    else:
        logger.warning("Sleep model file not found at %s", SLEEP_MODEL_PATH)

    if os.path.exists(MOOD_MODEL_PATH):
        mood_model = joblib.load(MOOD_MODEL_PATH)
        logger.info("Mood XGBoost model loaded from %s", MOOD_MODEL_PATH)
    else:
        logger.warning("Mood model file not found at %s", MOOD_MODEL_PATH)

    return sleep_model, mood_model


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_sleep_segment(data_segment, model):
    if model is None:
        return "Unknown"
    # data_segment: (1, Channels, 3000) → (1, 3000, Channels)
    if data_segment.shape[1] != 3000:
        data_segment = np.transpose(data_segment, (0, 2, 1))

    pred      = model.predict(data_segment, verbose=0)
    class_idx = np.argmax(pred, axis=1)[0]
    mapping   = {0: "Light", 1: "Deep", 2: "REM"}
    return mapping.get(class_idx, "Unknown")


def predict_mood_segment(ecg, gsr, model):
    if model is None:
        return "Unknown", {}

    hrv      = compute_hrv(ecg)
    gsr_feats = compute_gsr(gsr)
    feats     = np.array(hrv + gsr_feats).reshape(1, -1)

    cols = [
        "bpm", "sdnn", "rmssd", "pnn50", "lf", "lf_hf",
        "mean_gsr", "std_gsr", "num_peaks", "mean_peak_amp", "slope",
    ]
    df_feats = pd.DataFrame(feats, columns=cols)

    pred  = model.predict(df_feats)[0]
    probs = model.predict_proba(df_feats)[0]

    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping.get(pred, "Unknown"), {
        "Negative": float(probs[0]),
        "Neutral":  float(probs[1]),
        "Positive": float(probs[2]),
    }


# ── Heuristic simulation (no trained model required) ─────────────────────────
def simulate_sleep_logic(hr, movement, spo2, hrv_var, duration):
    """
    Rule-based sleep stage estimation from physiological parameters.
    Does NOT use the trained CNN-LSTM model.
    """
    score = 0
    if hr > 80 or movement > 7:       score += 1   # Light
    elif hr < 60 and movement < 3:    score -= 1   # Deep

    if hrv_var > 50 and movement < 2: score = 2    # REM bias

    if score >= 1:   stage = "Light"
    elif score <= -1: stage = "Deep"
    else:             stage = "REM"

    efficiency = max(0.0, min(100.0, 100 - (movement * 5) + (spo2 - 90)))

    return {
        "stage":         stage,
        "efficiency":    efficiency,
        "rem_percent":   20 + (hrv_var / 5),
        "deep_percent":  20 + ((100 - hr) / 2),
        "light_percent": 100 - (20 + (hrv_var / 5)) - (20 + ((100 - hr) / 2)),
    }


def simulate_mood_logic(rmssd, hr, gsr_peaks, gsr_slope, stress, mood_model=None):
    """
    Mood estimation from physiological parameters.
    Uses the trained XGBoost model if provided; otherwise falls back to heuristics.

    Parameters
    ----------
    mood_model : optional pre-loaded XGBoost model (avoids reloading from disk).
    """
    # Map simulation inputs to feature vector
    bpm          = hr
    sdnn         = rmssd * 1.2
    pnn50        = 10 if stress > 5 else 30
    lf           = 0.5
    lf_hf        = 2.0 if stress > 5 else 0.5
    mean_gsr     = stress * 2
    std_gsr      = 1.0
    num_peaks    = gsr_peaks
    mean_peak_amp = 0.5
    slope        = gsr_slope

    feats = np.array([[bpm, sdnn, rmssd, pnn50, lf, lf_hf,
                       mean_gsr, std_gsr, num_peaks, mean_peak_amp, slope]])
    cols = [
        "bpm", "sdnn", "rmssd", "pnn50", "lf", "lf_hf",
        "mean_gsr", "std_gsr", "num_peaks", "mean_peak_amp", "slope",
    ]
    df_feats = pd.DataFrame(feats, columns=cols)

    # Use passed model (avoids per-request disk reload)
    model = mood_model
    if model is None:
        # Last-resort fallback: try loading from disk once
        if os.path.exists(MOOD_MODEL_PATH):
            model = joblib.load(MOOD_MODEL_PATH)
            logger.warning("Mood model loaded inside simulate_mood_logic — pass it as argument for efficiency")

    if model is not None:
        pred  = model.predict(df_feats)[0]
        probs = model.predict_proba(df_feats)[0]
        mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return {
            "mood": mapping.get(pred, "Unknown"),
            "probabilities": {
                "Negative": float(probs[0]),
                "Neutral":  float(probs[1]),
                "Positive": float(probs[2]),
            },
        }

    # Pure heuristic fallback
    if stress > 6:
        mood, probs = "Negative", {"Negative": 0.7, "Neutral": 0.2, "Positive": 0.1}
    elif stress < 4:
        mood, probs = "Positive", {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}
    else:
        mood, probs = "Neutral",  {"Negative": 0.2, "Neutral": 0.6, "Positive": 0.2}
    return {"mood": mood, "probabilities": probs}


# ── Legacy suggestion helpers (kept for backward compat) ─────────────────────
def get_sleep_suggestions(efficiency, rem_percent, deep_percent):
    suggestions = []
    if efficiency < 85:
        suggestions.append("📉 **Low Sleep Efficiency**: Try to maintain a consistent sleep schedule.")
    if deep_percent < 15:
        suggestions.append("🧠 **Low Deep Sleep**: Ensure your room is cool and dark.")
    if rem_percent < 20:
        suggestions.append("👁️ **Low REM Sleep**: Reduce alcohol and manage stress.")
    if not suggestions:
        suggestions.append("✅ **Great Sleep!**: Keep up your healthy habits.")
    return suggestions


def get_mood_suggestions(mood, stress_level=None):
    suggestions = []
    if mood == "Negative":
        suggestions.append("🧘 **Stress Relief**: Try 5 minutes of deep breathing or meditation.")
        suggestions.append("📝 **Journaling**: Write down your worries to clear your mind.")
    elif mood == "Neutral":
        suggestions.append("⚖️ **Balance**: A short walk or listening to music might boost your mood.")
    else:
        suggestions.append("🌟 **Keep it up**: You're doing great! Share your positivity.")
    if stress_level and stress_level > 7:
        suggestions.append("⚠️ **High Stress Detected**: Consider taking a break or practising mindfulness.")
    return suggestions


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Running test inference…")
    s_res = simulate_sleep_logic(60, 1, 98, 60, 8)
    logger.info("Sleep Sim: %s", s_res)
    m_res = simulate_mood_logic(40, 70, 5, 0.1, 3)
    logger.info("Mood Sim: %s  probs: %s", m_res["mood"], m_res["probabilities"])
