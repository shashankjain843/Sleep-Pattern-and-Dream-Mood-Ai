# Sleep Pattern Detection & Dream Mood Prediction — Project Summary

**Version**: 2.0 (Production)  
**Last Updated**: July 2025  
**Status**: ✅ Fully Functional

---

## 📋 Project Overview

An end-to-end AI-powered system that analyses sleep patterns and predicts dream mood states from physiological signals. Features a futuristic neon-themed Streamlit dashboard, a FastAPI REST backend, a CNN-LSTM deep learning model for sleep staging, and an XGBoost classifier for mood prediction. Version 2 adds persistence, PDF export, authentication, model transparency, automated tests, Docker deployment, and optional Claude AI integration.

---

## 🏗️ System Architecture

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (Python) |
| Backend | FastAPI + Uvicorn |
| Sleep Model | TensorFlow/Keras CNN-LSTM |
| Mood Model | XGBoost |
| Signal Processing | MNE-Python, SciPy, NumPy |
| Persistence | SQLite (via stdlib `sqlite3`) |
| Auth | PyJWT + PBKDF2 password hashing |
| PDF Export | ReportLab |
| AI Narrative | Anthropic Claude (optional) |
| Tests | pytest + FastAPI TestClient |
| Deployment | Docker + Docker Compose |

### Architecture Flow

```
User (Browser)
    ↓
Streamlit Frontend (port 8501)
    ↓  HTTP  ↓
FastAPI Backend (port 8000)
    ├─ Preprocessing Layer (MNE / SciPy)
    ├─ ML Models: Sleep CNN-LSTM + Mood XGBoost
    ├─ Suggestions Engine (rule-based + optional Claude)
    ├─ SQLite Session Store
    └─ PDF Report Generator (ReportLab)
```

---

## 📊 Datasets

### 1. Sleep-EDF Database (PhysioNet)
- **Format**: European Data Format (.edf)
- **Channels used**: `EEG Fpz-Cz`, `EEG Pz-Oz` (2 channels)
- **Sampling Rate**: 100 Hz
- **Epoch Duration**: 30 seconds (3,000 samples/epoch)
- **Labels**: Sleep Stage W/1/2 → Light (0), Stage 3/4 → Deep (1), Stage R → REM (2)

### 2. YAAD ECG+GSR Dataset
- **Format**: CSV files (ECG and GSR signals)
- **Sampling Rate**: 100 Hz
- **Duration**: 30 seconds per trial (3,000 samples)
- **Labels**: valence score → Negative (0), Neutral (1), Positive (2)

---

## 🔧 Data Preprocessing

### Sleep (`preprocess_sleepedf.py`)
1. Load PSG EDF + hypnogram via MNE
2. Select 2 EEG channels (`EEG Fpz-Cz`, `EEG Pz-Oz`)
3. Bandpass filter: 0.5–30 Hz (firwin)
4. Segment into 30-second epochs
5. Z-score normalise per channel
6. Output shape: `(N, 2, 3000)`

### Mood (`preprocess_yaad.py` + `extract_features.py`)
1. Load ECG + GSR CSV files, resample to 3,000 samples
2. Stack into `(N, 2, 3000)` array
3. Extract HRV features: BPM, SDNN, RMSSD, pNN50, LF power, LF/HF ratio
4. Extract GSR features: mean, std, num_peaks, mean_peak_amplitude, slope
5. Combined feature vector: 11 features per sample → `features_yaad.csv`

---

## 🤖 Model Architecture

### Sleep Staging — CNN-LSTM (`train_sleep_model.py`)

**Actual architecture (corrected from earlier documentation):**

```
Input: (batch, 3000_timesteps, 2_channels)   ← transposed from (N, 2, 3000)
    ↓
Conv1D(64 filters, kernel=3, ReLU)
BatchNormalization()
MaxPooling1D(pool_size=2)
Dropout(0.3)
    ↓
Conv1D(128 filters, kernel=3, ReLU)
BatchNormalization()
MaxPooling1D(pool_size=2)
Dropout(0.3)
    ↓
LSTM(64 units)
Dropout(0.3)
    ↓
Dense(32, ReLU)
Dense(3, Softmax)
    ↓
Output: [Light, Deep, REM] probabilities
```

**Training:** Adam optimizer · sparse_categorical_crossentropy · 5 epochs · batch 32 · 80/20 split · class-balanced weights  
**Metrics saved**: `models/sleep_cnn_lstm/metrics.json` (accuracy, loss, classification report, confusion matrix)

### Mood Classification — XGBoost (`train_mood_model.py`)

```
Input: (N, 11_features)
    ↓
XGBClassifier(n_estimators=100, eval_metric='mlogloss')
    ↓
Output: [Negative, Neutral, Positive] label + probabilities
```

**Metrics saved**: `models/mood_xgb_metrics.json`

---

## 🚀 Inference Pipeline (`run_inference.py`)

### Model Inference
- **Sleep**: Load CNN-LSTM → preprocess EDF → predict per epoch → aggregate stats
- **Mood**: Load XGBoost → extract HRV+GSR features → predict label + probabilities

### Simulation Mode (heuristic — no trained model)
- **Sleep**: Rule-based: `HR + movement + SpO2 + HRV → stage + efficiency`
- **Mood**: Uses loaded XGBoost if available; falls back to stress-based heuristics

> **Important**: Simulation and model-based modes are clearly labelled in the UI. Sleep simulation is always heuristic. Mood simulation uses the trained model when loaded.

---

## 💡 Suggestions & Reports (`suggestions.py`)

### Rule-based suggestions (always available)
- Sleep: thresholds on REM%, deep%, efficiency, duration, awakenings
- Mood: label-specific suggestions (Positive / Neutral / Negative)

### Claude AI (optional, requires `ANTHROPIC_API_KEY`)
- **Narrative**: Personalised paragraph replacing fixed template sentences
- **Correlated suggestions**: Cross-metric pattern analysis (e.g. "high stress + low deep sleep")
- **Q&A**: Answer user questions grounded in their own session data

All Claude features degrade gracefully to rule-based output if the API key is not set or calls fail.

---

## 🔌 Backend API (`main_fastapi.py`)

### Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/status` | GET | ❌ | Real system status: TF available, models loaded, Claude available |
| `/register` | POST | ❌ | Register a new user |
| `/login` | POST | ❌ | Get JWT access token |
| `/history` | GET | Optional | Last 50 sessions for current user |
| `/predict_sleep_upload` | POST | Optional | Analyse uploaded .edf file |
| `/predict_sleep_sample` | POST | Optional | Analyse default Sleep-EDF sample |
| `/predict_mood_upload` | POST | Optional | Analyse uploaded ECG+GSR .csv |
| `/predict_mood_sample` | POST | Optional | Return sample mood prediction |
| `/simulate_sleep` | POST | Optional | Heuristic sleep estimation |
| `/simulate_mood` | POST | Optional | Model-based (or heuristic) mood estimation |
| `/generate_report` | POST | Optional | Full health report (JSON) |
| `/generate_report_pdf` | POST | Optional | Full health report (PDF download) |
| `/ask_insight` | POST | Optional | Claude-powered Q&A about session data |

All v1 endpoints remain backward-compatible.

---

## 🎨 User Interface (`streamlit_app.py`)

### Modes
1. **Real Input Mode** — Upload EDF/CSV files or use sample data; both labelled `🤖 Model-based prediction`
2. **Live Simulation Mode** — Slider-based: Sleep labelled `⚙️ Rule-based estimate`, Mood labelled dynamically based on whether the model is loaded
3. **📈 History & Trends** — Line charts of sleep efficiency, REM%, deep%, and mood over time (Plotly, neon theme)

### Features
- Real backend status in sidebar (TF, sleep model, mood model, Claude — all live from `/status`)
- Login / Register form in sidebar
- "🔬 About the Models" expander — shows accuracy % and training date from metrics JSON
- PDF download button in every report section
- Claude Q&A chat box in every report section
- `API_URL` configurable via environment variable (default: `http://localhost:8000`)

---

## 🗄️ Persistence (`database.py`)

SQLite database `sleep_ai.db` stores every analysis session:

```sql
sessions(id, timestamp, user_id, input_source,
         sleep_metrics_json, mood_metrics_json, suggestions_json)
```

Easy to migrate to Postgres by swapping `sqlite3` for `psycopg2`.

---

## 🔒 Authentication (`auth.py`)

- Passwords: PBKDF2-SHA256 + random salt (no bcrypt C extension needed)
- Tokens: JWT via PyJWT (pure Python, works on Python 3.14)
- User store: `users.json` flat file (swap for DB in production)
- Token expiry: 8 hours

---

## 📁 Project Structure

```
project/
├── datasets/                       # Raw data (not in version control)
│   ├── SC4001E0-PSG.edf
│   ├── SC4001EC-Hypnogram.edf
│   └── YAAD ECG+GSR dataset/
├── models/                         # Trained models + metrics
│   ├── sleep_cnn_lstm/
│   │   ├── model.h5
│   │   └── metrics.json            # NEW: accuracy, confusion matrix
│   ├── mood_xgb.model
│   └── mood_xgb_metrics.json       # NEW
├── tests/                          # NEW: pytest tests
│   ├── __init__.py
│   ├── test_extract_features.py
│   └── test_api.py
├── preprocess_sleepedf.py
├── preprocess_yaad.py
├── extract_features.py
├── train_sleep_model.py
├── train_mood_model.py
├── run_inference.py
├── suggestions.py
├── main_fastapi.py
├── streamlit_app.py
├── database.py                     # NEW: SQLite persistence
├── auth.py                         # NEW: JWT auth
├── llm_client.py                   # NEW: Claude integration
├── pdf_report.py                   # NEW: PDF generation
├── Dockerfile.backend              # NEW
├── Dockerfile.frontend             # NEW
├── docker-compose.yml              # NEW
├── .env.example                    # NEW
├── requirements.txt                # Pinned versions
├── run.bat                         # Windows launch script
└── PROJECT_SUMMARY.md
```

---

## 🔄 How to Run

### Local (development)

```powershell
# 1. Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure AI features
copy .env.example .env
# Edit .env and add ANTHROPIC_API_KEY

# 4. Start backend
python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000

# 5. Start frontend (new terminal)
python -m streamlit run streamlit_app.py

# 6. Open browser: http://localhost:8501
```

### Docker (production)

```bash
# Copy and configure .env
cp .env.example .env

# Build and start both services
docker compose up --build

# Access: http://localhost:8501
```

### Run tests

```powershell
pytest tests/ -v
```

---

## 🎯 Feature Matrix

| Feature | Status |
|---------|--------|
| CNN-LSTM sleep staging | ✅ |
| XGBoost mood classification | ✅ |
| HRV + GSR feature engineering | ✅ |
| Rule-based suggestions | ✅ |
| Neon Streamlit UI | ✅ |
| Real Input Mode | ✅ |
| Live Simulation Mode | ✅ |
| Mode labelling (model vs heuristic) | ✅ NEW |
| SQLite session persistence | ✅ NEW |
| History & Trends charts | ✅ NEW |
| PDF report export | ✅ NEW |
| JWT authentication | ✅ NEW |
| Real system status sidebar | ✅ NEW |
| Model accuracy metrics in UI | ✅ NEW |
| Structured logging (all files) | ✅ NEW |
| pytest test suite | ✅ NEW |
| Docker + Compose deployment | ✅ NEW |
| Claude AI narrative reports | ✅ NEW (opt-in) |
| Claude Q&A chat | ✅ NEW (opt-in) |
| Claude correlated suggestions | ✅ NEW (opt-in) |
| Backward-compatible API | ✅ |
