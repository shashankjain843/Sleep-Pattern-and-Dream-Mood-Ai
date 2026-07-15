# File: streamlit_app.py
"""
Sleep Pattern & Dream Mood AI — Streamlit frontend.

New in this version:
  - API_URL configurable via environment variable
  - Real sidebar status from /status endpoint
  - Mode badges: "⚙️ Rule-based estimate" vs "🤖 Model-based prediction"
  - History & Trends tab with Plotly charts
  - Download PDF Report button
  - Login / Register form in sidebar
  - "About the Models" expander with accuracy metrics
  - Claude-powered Q&A chat in report section
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

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

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Dream & Sleep AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌",
)

# ── Neon CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #111827 0%, #000000 100%);
        color: #e0e0e0;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0,255,204,0.7);
        letter-spacing: 1px;
    }
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #00ffcc;
        box-shadow: 5px 0 15px rgba(0,255,204,0.1);
    }
    .neon-card {
        background: rgba(20,20,30,0.6);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0,255,204,0.3);
        box-shadow: 0 0 15px rgba(0,255,204,0.2), inset 0 0 10px rgba(0,255,204,0.05);
        transition: all 0.3s ease;
        margin-bottom: 20px;
        text-align: center;
    }
    .neon-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 25px rgba(0,255,204,0.4);
        border-color: #00ffcc;
    }
    .neon-card-mood {
        border-color: rgba(255,0,255,0.3);
        box-shadow: 0 0 15px rgba(255,0,255,0.2);
    }
    .neon-card-mood:hover {
        box-shadow: 0 0 25px rgba(255,0,255,0.4);
        border-color: #ff00ff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ffcc 0%, #0099cc 100%);
        color: #000;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(0,255,204,0.5);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 20px rgba(0,255,204,0.8); color: #fff; }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00ffcc, #0099cc);
        box-shadow: 0 0 10px rgba(0,255,204,0.5);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 5px;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0,255,204,0.1) !important;
        border-color: #00ffcc !important;
        color: #00ffcc !important;
        box-shadow: 0 0 10px rgba(0,255,204,0.3);
    }
    .suggestion-box {
        background: rgba(10,10,15,0.9);
        border-left: 4px solid #00ffcc;
        padding: 15px;
        margin-top: 10px;
        border-radius: 0 10px 10px 0;
        color: #fff;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffcc, transparent);
        margin: 30px 0;
        box-shadow: 0 0 10px #00ffcc;
    }
    .badge-model {
        display: inline-block;
        background: rgba(0,255,204,0.15);
        color: #00ffcc;
        border: 1px solid #00ffcc;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-family: 'Orbitron', sans-serif;
        margin-bottom: 10px;
    }
    .badge-heuristic {
        display: inline-block;
        background: rgba(255,165,0,0.15);
        color: #ffa500;
        border: 1px solid #ffa500;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-family: 'Orbitron', sans-serif;
        margin-bottom: 10px;
    }
    .chat-answer {
        background: rgba(0,153,204,0.1);
        border-left: 4px solid #0099cc;
        padding: 14px;
        border-radius: 0 10px 10px 0;
        color: #e0e0e0;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    .status-dot-green  { color: #00ffcc; }
    .status-dot-orange { color: #ffa500; }
    .status-dot-red    { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def neon_card(title, value, suffix="", mood_variant=False):
    card_class = "neon-card neon-card-mood" if mood_variant else "neon-card"
    color = "#ff00ff" if mood_variant else "#00ffcc"
    return f"""
    <div class="{card_class}">
        <h3 style="color:#a0aec0;font-size:0.9rem;margin-bottom:5px;">{title}</h3>
        <h2 style="color:{color};font-size:2rem;margin:0;">{value}{suffix}</h2>
    </div>
    """


def model_badge(is_model: bool):
    if is_model:
        return "<span class='badge-model'>🤖 Model-based prediction</span>"
    return "<span class='badge-heuristic'>⚙️ Rule-based estimate</span>"


def auth_headers() -> dict:
    token = st.session_state.get("token", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    g_key = st.session_state.get("gemini_api_key", "")
    a_key = st.session_state.get("anthropic_api_key", "")
    if g_key:
        headers["X-Gemini-Key"] = g_key
    if a_key:
        headers["X-Anthropic-Key"] = a_key
    return headers


def api_get(path: str) -> dict:
    try:
        r = requests.get(f"{API_URL}{path}", headers=auth_headers(), timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_post(path: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", headers=auth_headers(), timeout=30, **kwargs)
        if r.status_code == 200:
            return r.json()
        st.error(f"Backend error {r.status_code}: {r.text[:200]}")
        return None
    except Exception as exc:
        st.error(f"Connection failed: {exc}")
        return None


# ── Fetch real backend status ─────────────────────────────────────────────────
def get_backend_status(gemini_key="", anthropic_key=""):
    try:
        headers = {}
        if gemini_key:
            headers["X-Gemini-Key"] = gemini_key
        if anthropic_key:
            headers["X-Anthropic-Key"] = anthropic_key
        r = requests.get(f"{API_URL}/status", headers=headers, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Main title ────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;margin-bottom:30px;'>🌌 SLEEP PATTERN & DREAM MOOD AI</h1>",
            unsafe_allow_html=True)
st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 NAVIGATION PANEL")
    mode = st.radio("SELECT MODE", ["Real Input Mode", "Live Simulation Mode", "📈 History & Trends", "🌌 Visual Dream Journal"])

    st.markdown("---")

    # ── Login / Register ──────────────────────────────────────────────────────
    st.markdown("### 👤 ACCOUNT")
    if st.session_state.get("username"):
        st.success(f"Logged in as **{st.session_state['username']}**")
        if st.button("Logout", key="btn_logout"):
            st.session_state.pop("token", None)
            st.session_state.pop("username", None)
            st.rerun()
    else:
        tab_login, tab_reg = st.tabs(["Login", "Register"])
        with tab_login:
            lu = st.text_input("Username", key="login_user")
            lp = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="btn_login"):
                resp = api_post("/login", json={"username": lu, "password": lp})
                if resp and "access_token" in resp:
                    st.session_state["token"]    = resp["access_token"]
                    st.session_state["username"] = resp["username"]
                    st.success(f"Welcome, {resp['username']}!")
                    st.rerun()
                elif resp:
                    st.error("Invalid credentials")
        with tab_reg:
            if st.session_state.get("verify_username"):
                username_verify = st.session_state["verify_username"]
                st.info(f"Verify registration for **{username_verify}**")
                
                mock_code = st.session_state.get("verify_mock_code", "")
                if mock_code:
                    st.success(f"📨 **[Mock Email Client]**\nCode: `{mock_code}` sent to your email.")
                
                verify_code = st.text_input("Enter 6-digit Code", key="input_verify_code")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    if st.button("Verify & Login", key="btn_verify_confirm"):
                        if not verify_code.strip():
                            st.error("Please enter the verification code.")
                        else:
                            resp = api_post("/verify_email", json={"username": username_verify, "verification_code": verify_code})
                            if resp and "access_token" in resp:
                                st.session_state["token"]    = resp["access_token"]
                                st.session_state["username"] = resp["username"]
                                st.session_state.pop("verify_username", None)
                                st.session_state.pop("verify_mock_code", None)
                                st.success("Registered and Logged in successfully!")
                                st.rerun()
                with v_col2:
                    if st.button("Cancel", key="btn_verify_cancel"):
                        st.session_state.pop("verify_username", None)
                        st.session_state.pop("verify_mock_code", None)
                        st.rerun()
            else:
                ru = st.text_input("Username", key="reg_user")
                re = st.text_input("Email", key="reg_email")
                rp = st.text_input("Password", type="password", key="reg_pass")
                if st.button("Register", key="btn_reg"):
                    if not ru.strip() or not re.strip() or not rp.strip():
                        st.error("All fields are required.")
                    else:
                        resp = api_post("/register", json={"username": ru, "password": rp, "email": re})
                        if resp and "mock_code" in resp:
                            st.session_state["verify_username"] = ru
                            st.session_state["verify_mock_code"] = resp["mock_code"]
                            st.rerun()




# ── Report helper (shared by both modes) ─────────────────────────────────────
def display_report_section(sleep_data: dict, mood_data: dict):
    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
    st.markdown("## 📋 YOUR PERSONALISED REPORT")

    st.markdown("### 📊 ADD DAILY HABITS (For AI Trend Analysis)")
    hc1, hc2 = st.columns(2)
    with hc1:
        caffeine_hour = st.number_input(
            "Last caffeine consumption hour (24-hour format, e.g. 15 for 3:00 PM)",
            min_value=0, max_value=23, value=15, step=1,
            help="Specify when you last consumed coffee, tea, or soda."
        )
    with hc2:
        workout_evening = st.checkbox(
            "Did you do a strenuous workout in the evening?",
            value=False,
            help="Checked if you worked out within 4 hours before bedtime."
        )

    with st.spinner("Analysing Bio-Metrics…"):
        sleep_metrics = {
            "efficiency":   sleep_data.get("efficiency", 0),
            "rem_percent":  sleep_data.get("rem_percent", 0),
            "deep_percent": sleep_data.get("deep_percent", 0),
            "light_percent":sleep_data.get("light_percent", 0),
            "duration":     sleep_data.get("duration", 8),
            "awakenings":   sleep_data.get("awakenings", 0),
            "caffeine_hour": caffeine_hour,
            "workout_evening": 1 if workout_evening else 0,
        }
        mood_metrics = {
            "mood":          mood_data.get("mood", "Unknown"),
            "probabilities": mood_data.get("probabilities", {}),
        }

        report = api_post("/generate_report", json={"sleep_metrics": sleep_metrics, "mood_metrics": mood_metrics})
        if not report:
            st.error("Report generation failed")
            return

    # ── AI indicator ──────────────────────────────────────────────────────────
    if report.get("claude_powered"):
        provider_name = report.get("ai_provider", "AI").capitalize()
        st.info(f"✨ **AI-Enhanced Report** — narrative generated by {provider_name} AI")

    st.markdown(f"<h2 style='text-align:center;color:#fff;'>{report['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:#ccc;'>{report['summary']}</p>", unsafe_allow_html=True)

    # ── Claude narrative (if present) ─────────────────────────────────────────
    if report.get("narrative"):
        st.markdown("### 🤖 AI Narrative")
        st.markdown(f"<div class='chat-answer'>{report['narrative']}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 💤 Sleep Summary")
        st.info(report["sleep_summary"].replace("**", ""))
    with c2:
        st.markdown("### 💭 Mood Summary")
        st.info(report["mood_summary"].replace("**", ""))

    st.markdown("### 🔗 Combined Insights")
    st.write(report["combined_insights"])

    st.markdown("### 💡 Personalised Recommendations")
    for s in report["suggestions_list"]:
        st.markdown(f"<div class='suggestion-box'>{s.replace('**','')}</div>", unsafe_allow_html=True)

    if report["warnings"]:
        st.markdown("### ⚠️ Warnings")
        for w in report["warnings"]:
            st.warning(w)

    # ── PDF download button ───────────────────────────────────────────────────
    st.markdown("---")
    if st.button("⬇️ Download PDF Report", key="btn_pdf"):
        with st.spinner("Generating PDF…"):
            try:
                r = requests.post(
                    f"{API_URL}/generate_report_pdf",
                    json={"sleep_metrics": sleep_metrics, "mood_metrics": mood_metrics},
                    headers=auth_headers(),
                    timeout=30,
                )
                if r.status_code == 200:
                    st.download_button(
                        label="📄 Click to save PDF",
                        data=r.content,
                        file_name="sleep_mood_report.pdf",
                        mime="application/pdf",
                        key="dl_pdf",
                    )
                elif r.status_code == 503:
                    st.warning("PDF generation requires ReportLab. Run: pip install reportlab")
                else:
                    st.error(f"PDF generation failed ({r.status_code})")
            except Exception as exc:
                st.error(f"PDF error: {exc}")

    # ── Claude Q&A chat ───────────────────────────────────────────────────────
    with st.expander("💬 Ask AI About Your Results"):
        session_ctx = {**sleep_metrics, **mood_metrics}
        question = st.text_input(
            "Ask a question about your sleep or mood data:",
            placeholder="e.g. Why is my deep sleep low?",
            key="insight_question",
        )
        if st.button("Ask", key="btn_ask"):
            if question.strip():
                with st.spinner("Thinking…"):
                    resp = api_post(
                        "/ask_insight",
                        json={"question": question, "session_metrics": session_ctx},
                    )
                if resp:
                    st.markdown(
                        f"<div class='chat-answer'>{resp.get('answer','No answer returned.')}</div>",
                        unsafe_allow_html=True,
                    )
                    src = resp.get("source", "")
                    if src == "claude":
                        st.caption("✨ Powered by Claude AI")
                    elif src == "unavailable":
                        st.caption("ℹ️ Set ANTHROPIC_API_KEY to enable AI answers")
            else:
                st.warning("Please enter a question.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Real Input Mode
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "Real Input Mode":
    st.markdown("## 📂 REAL DATA ANALYSIS")
    col1, col2 = st.columns(2)

    # ── Sleep section ─────────────────────────────────────────────────────────
    with col1:
        st.markdown("### 💤 SLEEP PATTERNS")
        st.markdown(model_badge(True), unsafe_allow_html=True)
        tab_sample, tab_upload = st.tabs(["USE SAMPLE", "UPLOAD EDF"])
        data_sleep = None

        with tab_sample:
            if st.button("USE REAL SLEEP SAMPLE", key="btn_real_sleep"):
                with st.spinner("Processing SC4001E0-PSG.edf…"):
                    data_sleep = api_post("/predict_sleep_sample")
                    if data_sleep:
                        st.session_state["data_sleep"] = data_sleep

        with tab_upload:
            uploaded_file = st.file_uploader("Upload .edf", type=["edf"])
            if uploaded_file and st.button("ANALYSE UPLOAD", key="btn_upload_sleep"):
                with st.spinner("Processing upload…"):
                    data_sleep = api_post(
                        "/predict_sleep_upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    )
                    if data_sleep:
                        st.session_state["data_sleep"] = data_sleep

        if data_sleep is None:
            data_sleep = st.session_state.get("data_sleep")

        if data_sleep and "error" not in data_sleep:
            stats  = data_sleep.get("stats", {})
            total  = len(data_sleep.get("timeline", []))
            rem_p  = data_sleep.get("rem_percent",  stats.get("REM",   0) / total * 100 if total else 0)
            deep_p = data_sleep.get("deep_percent", stats.get("Deep",  0) / total * 100 if total else 0)
            light_p= data_sleep.get("light_percent",stats.get("Light", 0) / total * 100 if total else 0)

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(neon_card("EFFICIENCY", data_sleep.get("efficiency", 0), "%"), unsafe_allow_html=True)
            with m2: st.markdown(neon_card("REM",   round(rem_p,  1), "%"), unsafe_allow_html=True)
            with m3: st.markdown(neon_card("DEEP",  round(deep_p, 1), "%"), unsafe_allow_html=True)
            with m4: st.markdown(neon_card("LIGHT", round(light_p,1), "%"), unsafe_allow_html=True)

            timeline = data_sleep.get("timeline", [])
            stage_map = {"Light": 0, "Deep": 1, "REM": 2}
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[stage_map.get(s, 0) for s in timeline],
                mode="lines+markers",
                name="Stage",
                line=dict(color="#00ffcc", width=2, shape="hv"),
                marker=dict(size=6, color="#00ffcc"),
            ))
            fig.update_layout(
                title="HYPNOGRAM", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(tickvals=[0,1,2], ticktext=["Light","Deep","REM"], gridcolor="#333"),
                xaxis=dict(gridcolor="#333", title="Epochs (30s)"),
                height=300, margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            if "suggestions" in data_sleep:
                st.markdown("#### 💡 AI INSIGHTS")
                for s in data_sleep["suggestions"]:
                    st.markdown(f"<div class='suggestion-box'>{s}</div>", unsafe_allow_html=True)

    # ── Mood section ──────────────────────────────────────────────────────────
    with col2:
        st.markdown("### 💭 DREAM MOOD")
        st.markdown(model_badge(True), unsafe_allow_html=True)
        tab_mood_sample, tab_mood_upload = st.tabs(["USE SAMPLE", "UPLOAD CSV"])
        data_mood = None

        with tab_mood_sample:
            if st.button("USE REAL EMOTION SAMPLE", key="btn_real_mood"):
                with st.spinner("Processing YAAD Trial…"):
                    data_mood = api_post("/predict_mood_sample")
                    if data_mood:
                        st.session_state["data_mood"] = data_mood

        with tab_mood_upload:
            uploaded_csv = st.file_uploader("Upload .csv", type=["csv"])
            if uploaded_csv and st.button("ANALYSE UPLOAD", key="btn_upload_mood"):
                with st.spinner("Processing upload…"):
                    data_mood = api_post(
                        "/predict_mood_upload",
                        files={"file": (uploaded_csv.name, uploaded_csv.getvalue())},
                    )
                    if data_mood:
                        st.session_state["data_mood"] = data_mood

        if data_mood is None:
            data_mood = st.session_state.get("data_mood")

        if data_mood and "error" not in data_mood:
            st.markdown(neon_card("PREDICTED MOOD", data_mood.get("mood","Unknown"), mood_variant=True), unsafe_allow_html=True)
            probs = data_mood.get("probabilities", {})
            fig_mood = go.Figure(data=[go.Bar(
                x=list(probs.keys()), y=list(probs.values()),
                marker_color=["#ff4b4b","#ffa500","#00ffcc"],
            )])
            fig_mood.update_layout(
                title="MOOD PROBABILITY DISTRIBUTION", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#333"), xaxis=dict(gridcolor="#333"),
                height=250, margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_mood, use_container_width=True)

            if "suggestions" in data_mood:
                st.markdown("#### 💡 EMOTIONAL GUIDANCE")
                for s in data_mood["suggestions"]:
                    st.markdown(f"<div class='suggestion-box' style='border-color:#ff00ff;'>{s}</div>", unsafe_allow_html=True)

    if st.session_state.get("data_sleep") and st.session_state.get("data_mood"):
        display_report_section(st.session_state["data_sleep"], st.session_state["data_mood"])


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Live Simulation Mode
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Live Simulation Mode":
    st.markdown("## 🎛️ LIVE SIMULATION")
    st.info(
        "**Note on simulation modes:** Sleep simulator uses **rule-based heuristics** "
        "(no trained model). Mood simulator uses the **trained XGBoost model** when loaded.",
        icon="ℹ️",
    )

    if "sim_sleep_data" not in st.session_state: st.session_state["sim_sleep_data"] = None
    if "sim_mood_data"  not in st.session_state: st.session_state["sim_mood_data"]  = None

    sim_col1, sim_col2 = st.columns(2)

    # ── Sleep simulator ───────────────────────────────────────────────────────
    with sim_col1:
        st.markdown("### 💤 SLEEP SIMULATOR")
        st.markdown(model_badge(False), unsafe_allow_html=True)  # heuristic badge
        with st.container():
            st.markdown("<div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:10px;'>", unsafe_allow_html=True)
            hr       = st.slider("Heart Rate (BPM)",   40, 120, 60, help="Average heart beats per minute during sleep.")
            movement = st.slider("Movement Level",      0,  10,  1, help="How much you move (0=Still, 10=Restless).")
            spo2     = st.slider("SpO2 (%)",           80, 100, 98, help="Blood oxygen level. 95-100% is normal.")
            hrv      = st.slider("HRV Variability",    10, 100, 50, help="Heart Rate Variability.")
            duration = st.slider("Duration (hrs)",      4,  10,  8, help="Total time spent sleeping.")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 SIMULATE SLEEP", key="btn_sim_sleep"):
            data = api_post("/simulate_sleep", json={"hr":hr,"movement":movement,"spo2":spo2,"hrv_var":hrv,"duration":duration})
            if data and "error" not in data:
                st.session_state["sim_sleep_data"] = data

        if st.session_state["sim_sleep_data"]:
            data = st.session_state["sim_sleep_data"]
            r1, r2 = st.columns(2)
            with r1: st.markdown(neon_card("STAGE",         data["stage"]),              unsafe_allow_html=True)
            with r2: st.markdown(neon_card("QUALITY SCORE", round(data["efficiency"],1), "%"), unsafe_allow_html=True)
            st.progress(min(data["efficiency"] / 100, 1.0))
            if "suggestions" in data:
                st.markdown("#### 💡 INSIGHTS")
                for s in data["suggestions"]:
                    st.markdown(f"<div class='suggestion-box'>{s}</div>", unsafe_allow_html=True)

    # ── Mood simulator ────────────────────────────────────────────────────────
    with sim_col2:
        st.markdown("### 💭 MOOD SIMULATOR")
        status = get_backend_status() or {}
        mood_loaded = status.get("mood_model_loaded", False)
        st.markdown(model_badge(mood_loaded), unsafe_allow_html=True)
        with st.container():
            st.markdown("<div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:10px;'>", unsafe_allow_html=True)
            rmssd  = st.slider("RMSSD (ms)",        10, 100,  40, help="Root Mean Square of Successive Differences.")
            hr_mood= st.slider("Heart Rate (Mood)", 50, 120,  70, help="Heart rate in dream state.")
            peaks  = st.slider("GSR Peaks",          0,  20,   5, help="Galvanic Skin Response peaks.")
            slope  = st.slider("GSR Slope",       -1.0, 1.0, 0.1, help="Trend of skin conductance.")
            stress = st.slider("Stress Level",       1,  10,   3, help="1=Calm, 10=Stressed.")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 SIMULATE DREAM MOOD", key="btn_sim_mood"):
            data = api_post("/simulate_mood", json={"rmssd":rmssd,"hr":hr_mood,"gsr_peaks":peaks,"gsr_slope":slope,"stress":stress})
            if data and "error" not in data:
                st.session_state["sim_mood_data"] = data

        if st.session_state["sim_mood_data"]:
            data = st.session_state["sim_mood_data"]
            st.markdown(neon_card("PREDICTED MOOD", data["mood"], mood_variant=True), unsafe_allow_html=True)
            probs = data["probabilities"]
            st.caption("Probability Distribution")
            for label, p in probs.items():
                st.text(f"{label}: {int(p*100)}%")
                st.progress(float(p))
            if "suggestions" in data:
                st.markdown("#### 💡 EMOTIONAL GUIDANCE")
                for s in data["suggestions"]:
                    st.markdown(f"<div class='suggestion-box' style='border-color:#ff00ff;'>{s}</div>", unsafe_allow_html=True)

    if st.session_state["sim_sleep_data"] and st.session_state["sim_mood_data"]:
        display_report_section(st.session_state["sim_sleep_data"], st.session_state["sim_mood_data"])


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — History & Trends
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 History & Trends":
    st.markdown("## 📈 HISTORY & TRENDS")

    if not st.session_state.get("username"):
        st.warning("Log in to see your personal history. Anonymous sessions are stored under 'anonymous'.")

    with st.spinner("Loading history…"):
        resp = api_get("/history")

    sessions = resp.get("sessions", [])
    user     = resp.get("user", "anonymous")

    if not sessions:
        st.info(f"No sessions found for **{user}**. Run an analysis first!")
    else:
        st.caption(f"Showing last {len(sessions)} sessions for **{user}**")

        # Build a DataFrame for charting
        rows = []
        for s in reversed(sessions):  # oldest first for timeline
            sm = s.get("sleep_metrics", {})
            mm = s.get("mood_metrics",  {})
            mood_map = {"Positive": 2, "Neutral": 1, "Negative": 0, "Unknown": 1}
            rows.append({
                "timestamp":  s["timestamp"][:16].replace("T", " "),
                "efficiency": sm.get("efficiency", sm.get("sleep_efficiency", None)),
                "rem_%":      sm.get("rem_percent",  None),
                "deep_%":     sm.get("deep_percent", None),
                "mood_score": mood_map.get(mm.get("mood", "Unknown"), 1),
                "mood_label": mm.get("mood", "–"),
                "source":     s.get("input_source", "–"),
            })

        df = pd.DataFrame(rows)
        df_with_sleep = df[df["efficiency"].notna()]
        df_with_mood  = df[df["mood_score"].notna()]

        # ── Sleep efficiency trend ────────────────────────────────────────────
        if not df_with_sleep.empty:
            st.markdown("### 💤 Sleep Efficiency Over Time")
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(
                x=df_with_sleep["timestamp"], y=df_with_sleep["efficiency"],
                mode="lines+markers", name="Efficiency %",
                line=dict(color="#00ffcc", width=2),
                marker=dict(size=8, color="#00ffcc"),
            ))
            if "rem_%" in df_with_sleep.columns:
                fig_eff.add_trace(go.Scatter(
                    x=df_with_sleep["timestamp"], y=df_with_sleep["rem_%"],
                    mode="lines+markers", name="REM %",
                    line=dict(color="#0099cc", width=2, dash="dot"),
                    marker=dict(size=6),
                ))
                fig_eff.add_trace(go.Scatter(
                    x=df_with_sleep["timestamp"], y=df_with_sleep["deep_%"],
                    mode="lines+markers", name="Deep %",
                    line=dict(color="#9966ff", width=2, dash="dash"),
                    marker=dict(size=6),
                ))
            fig_eff.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 100], gridcolor="#333", title="%"),
                xaxis=dict(gridcolor="#333"),
                height=320, margin=dict(l=20,r=20,t=20,b=40),
                legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        # ── Mood trend ────────────────────────────────────────────────────────
        if not df_with_mood.empty:
            st.markdown("### 💭 Dream Mood Over Time")
            color_map = {2: "#00ffcc", 1: "#ffa500", 0: "#ff4b4b"}
            colors = [color_map.get(s, "#aaa") for s in df_with_mood["mood_score"]]
            fig_mood = go.Figure()
            fig_mood.add_trace(go.Scatter(
                x=df_with_mood["timestamp"], y=df_with_mood["mood_score"],
                mode="lines+markers", name="Mood",
                line=dict(color="#ff00ff", width=2),
                marker=dict(size=10, color=colors),
                text=df_with_mood["mood_label"],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig_mood.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(
                    tickvals=[0,1,2], ticktext=["Negative","Neutral","Positive"],
                    gridcolor="#333",
                ),
                xaxis=dict(gridcolor="#333"),
                height=280, margin=dict(l=20,r=20,t=20,b=40),
            )
            st.plotly_chart(fig_mood, use_container_width=True)

        # ── Raw session table ─────────────────────────────────────────────────
        with st.expander("🗂️ Raw Session Log"):
            st.dataframe(
                df[["timestamp","source","efficiency","rem_%","deep_%","mood_label"]],
                use_container_width=True,
            )

        st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 AI SLEEP CYCLE OPTIMIZER & TREND ANALYST")
        st.markdown(
            "This engine runs correlations across your sleep history (caffeine intake times, "
            "evening workouts, sleep efficiency) and uses AI to optimize your circadian rhythm."
        )
        
        if st.button("🔬 RUN AI DATA ANALYSIS", key="btn_run_analysis"):
            with st.spinner("Analyzing database history and correlating metrics…"):
                analysis = api_get("/analyze_trends")
                
                if analysis and "error" not in analysis:
                    # Render correlations
                    corrs = analysis.get("correlations", {})
                    if corrs:
                        st.markdown("#### 📊 Discovered Bio-Habit Correlations")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if "caffeine_vs_efficiency" in corrs:
                                val = corrs["caffeine_vs_efficiency"]
                                st.metric("Caffeine Hour vs Sleep Efficiency", f"{val}", help="Negative value indicates later caffeine lowers efficiency.")
                        with cc2:
                            if "workout_vs_deep_sleep" in corrs:
                                val = corrs["workout_vs_deep_sleep"]
                                st.metric("Evening Workout vs Deep Sleep %", f"{val}", help="Negative value indicates evening workouts reduce deep sleep.")
                    
                    # Render rule-based flags
                    flags = analysis.get("flags", [])
                    if flags:
                        st.markdown("#### 🚨 Circadian Cycle Flags")
                        for f in flags:
                            st.warning(f)
                            
                    # Render AI advice
                    st.markdown("#### 💡 AI Circadian Cutoffs & Personalised Tip")
                    st.markdown(f"<div class='chat-answer'>{analysis.get('personalized_report')}</div>", unsafe_allow_html=True)
                    st.caption("✨ Powered by Google Gemini AI")
                else:
                    err_msg = analysis.get("error", "Trend analysis failed.") if analysis else "Trend analysis failed."
                    st.error(f"{err_msg} Log at least 2 sessions with caffeine/workout habits to compute trends.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 4 — Visual Dream Journal
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "🌌 Visual Dream Journal":
    st.markdown("## 🌌 VISUAL DREAM JOURNAL")
    st.markdown(
        "AI will psychologically interpret your dream based on last night's "
        "biometrics, and dynamically paint a visual art of your dream!"
    )
    
    # Try fetching biometric data from the latest session
    latest_rem = 20.0
    latest_hr = 70
    latest_mood = "Neutral"
    
    resp = api_get("/history")
    sessions = resp.get("sessions", [])
    if sessions:
        latest = sessions[0]
        sm = latest.get("sleep_metrics", {})
        mm = latest.get("mood_metrics", {})
        latest_rem = sm.get("rem_percent", latest_rem)
        latest_hr = int(sm.get("hr", mm.get("hr", latest_hr)))
        latest_mood = mm.get("mood", latest_mood)
        st.info(f"📋 Associated last session biometrics: **REM Sleep {latest_rem}%**, **HR {latest_hr} BPM**, **Predicted Mood: {latest_mood}**.")
    else:
        st.warning("⚠️ No sleep logs found in history. Using standard default biometrics.")
        
    with st.form("dream_form"):
        dream_text = st.text_area("What did you dream about?", 
                                  placeholder="e.g. I was flying through a neon city, and then it started raining stars...",
                                  height=150)
        
        st.markdown("**Adjust Biometrics manually if desired:**")
        b1, b2, b3 = st.columns(3)
        with b1:
            rem_val = st.slider("REM Sleep %", 0.0, 50.0, float(latest_rem))
        with b2:
            hr_val = st.slider("Sleep Heart Rate (BPM)", 40, 120, int(latest_hr))
        with b3:
            mood_val = st.selectbox("Dream Mood", ["Positive", "Neutral", "Negative"], index=["Positive", "Neutral", "Negative"].index(latest_mood) if latest_mood in ["Positive", "Neutral", "Negative"] else 1)
            
        submit_dream = st.form_submit_button("🔮 ANALYZE & PAINT DREAM")
        
    if submit_dream:
        if not dream_text.strip():
            st.error("Please describe your dream first!")
        else:
            with st.spinner("Analyzing dream symbols and painting visual scene…"):
                payload = {
                    "dream_text": dream_text,
                    "mood": mood_val,
                    "rem_percent": rem_val,
                    "heart_rate": hr_val
                }
                res = api_post("/analyze_dream", json=payload)
                
                if res and "error" not in res:
                    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
                    
                    dc1, dc2 = st.columns([3, 2])
                    with dc1:
                        st.markdown(f"### 🏷️ Mood Tag: **{res.get('mood_tag', 'Dream')}**")
                        st.markdown("### 🧠 Psychological Interpretation")
                        st.markdown(f"<div class='chat-answer' style='font-size:1.1rem; border-left-color: #ff00ff;'>{res.get('interpretation')}</div>", unsafe_allow_html=True)
                        st.caption("✨ Powered by Google Gemini AI")
                        
                        st.markdown("#### 🎨 Generated Prompt for Art:")
                        st.caption(res.get("image_prompt"))
                    with dc2:
                        st.markdown("### 🖼️ Painted Dream Scene")
                        st.image(res.get("image_url"), caption=f"Dream visual matching tag: {res.get('mood_tag')}", use_container_width=True)
                else:
                    st.error("Could not complete dream analysis. Verify your Gemini key settings in the sidebar.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#555;'>© 2025 Sleep & Dream AI | Futuristic Health Tech</div>",
    unsafe_allow_html=True,
)
