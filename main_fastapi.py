# File: main_fastapi.py
"""
FastAPI backend for Sleep Pattern & Dream Mood AI.

New in this version:
  - Proper logging throughout
  - SQLite session persistence (/history endpoint)
  - PDF report export (/generate_report_pdf)
  - JWT authentication (/login, /register)
  - Real system status endpoint (/status)
  - /ask_insight endpoint for Claude Q&A
  - simulate_mood_logic now uses pre-loaded model (fixes per-request reload bug)
"""
import logging
import os
import shutil
import tempfile
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Load .env file manually to avoid dependency on python-dotenv on Python 3.14
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env_file()

from run_inference import (
    load_models, predict_sleep_segment, predict_mood_segment,
    simulate_sleep_logic, simulate_mood_logic, TF_AVAILABLE,
)
from preprocess_sleepedf import load_and_preprocess_file
from preprocess_yaad import load_single_yaad_trial
from suggestions import generate_full_report, generate_sleep_suggestions, generate_mood_suggestions
from database import init_db, save_session, get_sessions
from auth import register_user, authenticate_user, create_access_token, get_current_user, _load_users
import llm_client
import dream_mood_module
import sleep_pattern_analyst

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sleep Pattern & Dream Mood AI",
    description="Production-ready backend for sleep staging and mood prediction.",
    version="2.0.0",
)

# ── Global models (loaded once at startup) ────────────────────────────────────
sleep_model, mood_model = load_models()
init_db()
logger.info(
    "Startup complete | TF_AVAILABLE=%s | sleep_model=%s | mood_model=%s",
    TF_AVAILABLE,
    sleep_model is not None,
    mood_model is not None,
)


# ── Request / Response models ─────────────────────────────────────────────────
class SimulationInputSleep(BaseModel):
    hr:       float
    movement: float
    spo2:     float
    hrv_var:  float
    duration: float


class SimulationInputMood(BaseModel):
    rmssd:     float
    hr:        float
    gsr_peaks: float
    gsr_slope: float
    stress:    float


class ReportRequest(BaseModel):
    sleep_metrics: dict
    mood_metrics:  dict


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    email:    Optional[str] = None


class VerifyRequest(BaseModel):
    username:          str
    verification_code: str


class InsightRequest(BaseModel):
    question:        str
    session_metrics: dict


class DreamRequest(BaseModel):
    dream_text:  str
    mood:        str
    rem_percent: float
    heart_rate:  int


# ── Helper ────────────────────────────────────────────────────────────────────
def _user_from_header(authorization: Optional[str] = None) -> str:
    """Extract username from auth header, or return 'anonymous'."""
    user = get_current_user(authorization)
    return user if user else "anonymous"


def _np_to_python(obj):
    """Recursively convert numpy types to native Python types for JSON safety."""
    if isinstance(obj, dict):
        return {k: _np_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_np_to_python(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    return obj


# ── Auth endpoints ────────────────────────────────────────────────────────────
pending_registrations = {}  # key: username, value: {"password": "...", "email": "...", "code": "..."}

@app.post("/register")
async def api_register(req: RegisterRequest):
    username = req.username.strip()

    if not username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    # If email is not provided, register directly (for backwards compatibility & tests)
    if not req.email or not req.email.strip():
        success = register_user(username, req.password, f"{username}@example.com")
        if not success:
            raise HTTPException(status_code=409, detail="Username already taken.")
        return {"message": f"User '{username}' registered successfully."}

    email = req.email.strip()

    # 1. Check existing in users.json
    users = _load_users()
    if username in users:
        raise HTTPException(status_code=409, detail="Username already registered. Please login instead.")
    for u, data in users.items():
        if data.get("email") == email:
            raise HTTPException(status_code=409, detail="Email already registered. Please login instead.")

    # 2. Check pending registrations
    for u, data in pending_registrations.items():
        if data.get("email") == email and u != username:
            raise HTTPException(status_code=409, detail="Email registration is already pending.")

    # Generate a random 6-digit verification code
    import random
    code = f"{random.randint(100000, 999999)}"
    pending_registrations[username] = {
        "password": req.password,
        "email": email,
        "code": code
    }

    logger.info("Mock verification code sent to %s: %s", email, code)
    return {
        "message": f"Verification code sent to {email}.",
        "mock_code": code
    }


@app.post("/verify_email")
async def api_verify_email(req: VerifyRequest):
    username = req.username.strip()
    code = req.verification_code.strip()

    if username not in pending_registrations:
        raise HTTPException(status_code=400, detail="No pending registration found for this username.")

    pending = pending_registrations[username]
    if pending["code"] != code:
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    # Save user to DB (users.json)
    success = register_user(username, pending["password"], pending["email"])
    if not success:
        raise HTTPException(status_code=409, detail="Failed to register user. Username or email may have been taken.")

    # Clear pending entry
    pending_registrations.pop(username)

    # Automatically generate access token to log them in directly
    token = create_access_token(username)
    if token is None:
        raise HTTPException(status_code=500, detail="Token generation failed (PyJWT not installed).")

    return {
        "access_token": token,
        "username": username,
        "message": "Email verified and logged in successfully!"
    }


@app.post("/login")
async def api_login(req: LoginRequest):
    if not authenticate_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_access_token(req.username)
    if token is None:
        raise HTTPException(status_code=500, detail="Token generation failed (PyJWT not installed).")
    return {"access_token": token, "token_type": "bearer", "username": req.username}


# ── Status endpoint ───────────────────────────────────────────────────────────
@app.get("/status")
async def api_status(
    x_gemini_key: Optional[str] = Header(default=None),
    x_anthropic_key: Optional[str] = Header(default=None),
):
    sleep_loaded = (sleep_model is not None) or os.path.exists("models/sleep_rf.model")
    gemini_avail = llm_client.is_available(x_gemini_key, "gemini")
    claude_avail = llm_client.is_available(x_anthropic_key, "claude")
    return {
        "tf_available":       TF_AVAILABLE,
        "sleep_model_loaded": sleep_loaded,
        "mood_model_loaded":  mood_model is not None,
        "claude_available":   claude_avail or gemini_avail,
        "gemini_available":   gemini_avail,
    }


# ── History endpoint ──────────────────────────────────────────────────────────
@app.get("/history")
async def api_history(authorization: Optional[str] = Header(default=None)):
    user = _user_from_header(authorization)
    sessions = get_sessions(user_id=user)
    return {"user": user, "sessions": sessions}


# ── Report endpoints ──────────────────────────────────────────────────────────
@app.post("/generate_report")
async def api_generate_report(
    req: ReportRequest,
    authorization: Optional[str] = Header(default=None),
    x_gemini_key: Optional[str] = Header(default=None),
    x_anthropic_key: Optional[str] = Header(default=None),
):
    user = _user_from_header(authorization)
    s_metrics = req.sleep_metrics.copy()
    report = generate_full_report(
        s_metrics, 
        req.mood_metrics, 
        gemini_key=x_gemini_key, 
        anthropic_key=x_anthropic_key
    )

    try:
        save_session(
            user_id=user,
            input_source="combined_report",
            sleep_metrics=s_metrics,
            mood_metrics=req.mood_metrics,
            suggestions=report["suggestions_list"],
        )
    except Exception:
        logger.warning("Failed to save combined session", exc_info=True)

    return report


@app.post("/generate_report_pdf")
async def api_generate_report_pdf(
    req: ReportRequest,
    authorization: Optional[str] = Header(default=None),
    x_gemini_key: Optional[str] = Header(default=None),
    x_anthropic_key: Optional[str] = Header(default=None),
):
    from pdf_report import generate_report_pdf
    user = _user_from_header(authorization)
    s_metrics = req.sleep_metrics.copy()
    report    = generate_full_report(
        s_metrics, 
        req.mood_metrics, 
        gemini_key=x_gemini_key, 
        anthropic_key=x_anthropic_key
    )
    pdf_bytes = generate_report_pdf(report)
    if pdf_bytes is None:
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable — install reportlab: pip install reportlab",
        )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=sleep_mood_report.pdf"},
    )


# ── LLM Q&A endpoint ─────────────────────────────────────────────────────────
@app.post("/ask_insight")
async def api_ask_insight(
    req: InsightRequest,
    x_gemini_key: Optional[str] = Header(default=None),
    x_anthropic_key: Optional[str] = Header(default=None),
):
    # Determine provider and key
    api_key = x_gemini_key
    provider = "gemini"
    if not api_key:
        api_key = x_anthropic_key
        provider = "claude"
    if not api_key:
        # Fallback to backend env variables
        if os.environ.get("GEMINI_API_KEY"):
            api_key = os.environ.get("GEMINI_API_KEY")
            provider = "gemini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            provider = "claude"

    if not llm_client.is_available(api_key, provider):
        return {
            "answer": (
                "AI insights require a Gemini or Anthropic API key. "
                "Enter it in the sidebar API Settings to enable this feature."
            ),
            "source": "unavailable",
        }
    answer = llm_client.answer_insight(req.question, req.session_metrics, api_key=api_key, provider=provider)
    if answer is None:
        return {"answer": f"Could not get an answer from {provider.capitalize()}. Please try again.", "source": "error"}
    return {"answer": answer, "source": provider}


# ── Dream Journal endpoint ───────────────────────────────────────────────────
@app.post("/analyze_dream")
async def api_analyze_dream(
    req: DreamRequest,
    x_gemini_key: Optional[str] = Header(default=None),
):
    result = dream_mood_module.process_dream_entry(
        dream_text=req.dream_text,
        mood=req.mood,
        rem_percent=req.rem_percent,
        heart_rate=req.heart_rate,
        api_key=x_gemini_key
    )
    return result


# ── AI Trend Analyst endpoint ─────────────────────────────────────────────────
@app.get("/analyze_trends")
async def api_analyze_trends(
    authorization: Optional[str] = Header(default=None),
    x_gemini_key: Optional[str] = Header(default=None),
):
    user = _user_from_header(authorization)
    result = sleep_pattern_analyst.run_full_analysis(
        user_id=user,
        api_key=x_gemini_key
    )
    return result


# ── Sleep prediction endpoints ────────────────────────────────────────────────
@app.post("/predict_sleep_upload")
async def api_predict_sleep_upload(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    user = _user_from_header(authorization)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        X, _ = load_and_preprocess_file(tmp_path, hyp_path=None)
        if X is None:
            return {"error": "Could not process EDF file. Ensure it is a valid PSG/EDF."}

        predictions = []
        for i in range(min(20, len(X))):
            seg  = X[i: i + 1]
            pred = predict_sleep_segment(seg, sleep_model)
            predictions.append(pred)

        unique, counts = np.unique(predictions, return_counts=True)
        stats = dict(zip(unique, counts.tolist()))

        deep_count = stats.get("Deep", 0)
        rem_count  = stats.get("REM", 0)
        total      = len(predictions)
        efficiency = 70 + (deep_count / total * 20) if total > 0 else 0

        metrics = {
            "rem_percent":  (rem_count / total) * 100 if total else 0,
            "deep_percent": (deep_count / total) * 100 if total else 0,
            "efficiency":   round(efficiency, 1),
            "duration":     8,
        }
        suggestions = generate_sleep_suggestions(metrics)

        result = _np_to_python({
            "timeline":    predictions,
            "stats":       stats,
            "efficiency":  metrics["efficiency"],
            "rem_percent": metrics["rem_percent"],
            "deep_percent":metrics["deep_percent"],
            "suggestions": suggestions,
            "source":      "model" if sleep_model is not None else "unavailable",
        })

        # Persist session
        try:
            save_session(
                user_id=user,
                input_source="upload_edf",
                sleep_metrics=metrics,
                mood_metrics={},
                suggestions=suggestions,
            )
        except Exception:
            logger.warning("Failed to save session", exc_info=True)

        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict_sleep_sample")
async def api_predict_sleep_sample(authorization: Optional[str] = Header(default=None)):
    user = _user_from_header(authorization)
    psg_path = "datasets/SC4001E0-PSG.edf"
    hyp_path = "datasets/SC4001EC-Hypnogram.edf"

    if not os.path.exists(psg_path):
        return {"error": "Sample file not found. Please upload a file instead."}

    X, y = load_and_preprocess_file(psg_path, hyp_path)
    if X is None:
        return {"error": "Failed to process sample"}

    predictions = []
    for i in range(min(10, len(X))):
        seg  = X[i: i + 1]
        pred = predict_sleep_segment(seg, sleep_model)
        predictions.append(pred)

    unique, counts = np.unique(predictions, return_counts=True)
    stats      = dict(zip(unique, counts.tolist()))
    efficiency = 75.0
    metrics    = {"rem_percent": 20, "deep_percent": 20, "efficiency": efficiency, "duration": 8}
    suggestions = generate_sleep_suggestions(metrics)

    result = _np_to_python({
        "timeline":    predictions,
        "stats":       stats,
        "efficiency":  efficiency,
        "rem_percent": 20,
        "deep_percent":20,
        "suggestions": suggestions,
        "source":      "model" if sleep_model is not None else "unavailable",
    })

    try:
        save_session(user, "sample_edf", metrics, {}, suggestions)
    except Exception:
        logger.warning("Failed to save session", exc_info=True)

    return result


# ── Mood prediction endpoints ─────────────────────────────────────────────────
@app.post("/predict_mood_upload")
async def api_predict_mood_upload(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    user = _user_from_header(authorization)
    try:
        df = pd.read_csv(file.file)
        if "ecg" not in df.columns or "gsr" not in df.columns:
            return {"error": "CSV must contain 'ecg' and 'gsr' columns."}

        ecg = df["ecg"].values
        gsr = df["gsr"].values

        from scipy.signal import resample
        if len(ecg) != 3000: ecg = resample(ecg, 3000)
        if len(gsr) != 3000: gsr = resample(gsr, 3000)

        mood, probs = predict_mood_segment(ecg, gsr, mood_model)
        suggestions = generate_mood_suggestions(mood, probs)

        result = {
            "mood":        mood,
            "probabilities": probs,
            "suggestions": suggestions,
            "source":      "model" if mood_model is not None else "heuristic",
        }

        try:
            save_session(user, "upload_csv", {}, {"mood": mood, "probabilities": probs}, suggestions)
        except Exception:
            logger.warning("Failed to save session", exc_info=True)

        return result
    except Exception as exc:
        logger.exception("Mood upload prediction failed")
        return {"error": f"Failed to process CSV: {exc}"}


@app.post("/predict_mood_sample")
async def api_predict_mood_sample(authorization: Optional[str] = Header(default=None)):
    user = _user_from_header(authorization)
    try:
        mood        = "Positive"
        probs       = {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}
        suggestions = generate_mood_suggestions(mood, probs)

        result = {
            "mood":        mood,
            "probabilities": probs,
            "suggestions": suggestions,
            "source":      "sample",
        }

        try:
            save_session(user, "sample_mood", {}, {"mood": mood, "probabilities": probs}, suggestions)
        except Exception:
            logger.warning("Failed to save session", exc_info=True)

        return result
    except Exception as exc:
        logger.exception("Mood sample prediction failed")
        return {"error": str(exc)}


# ── Simulation endpoints ───────────────────────────────────────────────────────
@app.post("/simulate_sleep")
async def api_simulate_sleep(
    input: SimulationInputSleep,
    authorization: Optional[str] = Header(default=None),
):
    user = _user_from_header(authorization)
    try:
        res = simulate_sleep_logic(
            input.hr, input.movement, input.spo2, input.hrv_var, input.duration
        )
        res = _np_to_python(res)

        metrics = {
            "rem_percent":  res["rem_percent"],
            "deep_percent": res["deep_percent"],
            "efficiency":   res["efficiency"],
            "duration":     input.duration,
        }
        res["suggestions"] = generate_sleep_suggestions(metrics)
        res["source"]      = "heuristic"   # ← clear label: NOT the trained model

        try:
            save_session(user, "simulate_sleep", metrics, {}, res["suggestions"])
        except Exception:
            logger.warning("Failed to save session", exc_info=True)

        return res
    except Exception as exc:
        import traceback
        logger.exception("simulate_sleep failed")
        return {"error": str(exc)}


@app.post("/simulate_mood")
async def api_simulate_mood(
    input: SimulationInputMood,
    authorization: Optional[str] = Header(default=None),
):
    user = _user_from_header(authorization)
    try:
        # Pass pre-loaded mood_model to avoid per-request disk reload (bug fix)
        res = simulate_mood_logic(
            input.rmssd, input.hr, input.gsr_peaks, input.gsr_slope, input.stress,
            mood_model=mood_model,
        )

        if "probabilities" in res:
            res["probabilities"] = {k: float(v) for k, v in res["probabilities"].items()}

        res["suggestions"] = generate_mood_suggestions(res["mood"], res["probabilities"])
        res["source"]      = "model" if mood_model is not None else "heuristic"

        try:
            save_session(
                user, "simulate_mood", {},
                {"mood": res["mood"], "probabilities": res["probabilities"]},
                res["suggestions"],
            )
        except Exception:
            logger.warning("Failed to save session", exc_info=True)

        return res
    except Exception as exc:
        logger.exception("simulate_mood failed")
        return {"error": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
