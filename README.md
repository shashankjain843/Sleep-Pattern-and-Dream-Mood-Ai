# Sleep Pattern Detection and Dream Mood Prediction — Intelligent AI System

An end-to-end, AI-powered system designed to analyze sleep patterns and predict dream mood states from physiological signals. This system comprises a futuristic neon-themed **Streamlit dashboard** frontend, a robust **FastAPI REST backend**, a deep learning **CNN-LSTM model** for sleep staging, and an **XGBoost classifier** for mood prediction.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Core Features](#-core-features)
3. [Technology Stack](#%EF%B8%8F-technology-stack)
4. [System Architecture](#%EF%B8%8F-system-architecture)
5. [Project Structure](#-project-structure)
6. [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
7. [AI Models & Training](#-ai-models--training)
8. [API Endpoints](#-api-endpoints)
9. [Getting Started (Local Installation)](#-getting-started-local-installation)
10. [Docker Deployment](#-docker-deployment)
11. [Running Tests](#-running-tests)

---

## 📋 Project Overview

The **Sleep Pattern Detection and Dream Mood Prediction** system uses advanced signal processing and machine learning to analyze raw physiological signals (EEG, ECG, and GSR) to provide insights into sleep stages and dream mood states. 

- **Sleep Staging**: Processes European Data Format (.edf) sleep signals (EEG) to detect sleep stages (Light, Deep, and REM).
- **Dream Mood Prediction**: Processes ECG + GSR data to extract Heart Rate Variability (HRV) and Galvanic Skin Response (GSR) features, feeding them into a gradient boosted classifier to predict emotional valence (Positive, Neutral, Negative).
- **AI-Powered Insights**: Integrates optionally with Anthropic's Claude to generate narrative reports, correlated recommendations, and an interactive chat assistant grounded in session data.

---

## 🚀 Core Features

- **Double Model-Based Prediction**:
  - **CNN-LSTM Network** for sleep staging (trained on PhysioNet Sleep-EDF).
  - **XGBoost Classifier** for mood estimation (trained on YAAD ECG+GSR dataset).
- **Futuristic Streamlit Dashboard**: Neon-themed interface featuring real-time data plots (Plotly), historical trends, authentication forms, system status monitoring, and interactive expanders displaying model metrics.
- **FastAPI Backend Server**: Fully asynchronous endpoints supporting EDF/CSV file uploads, simulated estimations, user authentication, and chat integrations.
- **SQLite Session Persistence**: Persists user session records containing timestamps, inputs, predictions, and recommendations.
- **JWT-Based Authentication**: Secure registration and token-based login flow.
- **PDF Report Generator**: Generates and exports downloadable clinical-style PDF reports using ReportLab.
- **Claude AI Integration (Optional)**: Provides personalized text summaries, cross-metric recommendations, and a grounded Q&A chat based on sleep session results.
- **Dockerized Architecture**: Simplified multi-container deployment via Docker and Docker Compose.
- **Comprehensive Test Suite**: Automated endpoint and feature extraction testing using `pytest`.

---

## 🏗️ Technology Stack

| Layer / Component | Technology Used |
| :--- | :--- |
| **Frontend** | Streamlit, Plotly, HTML/CSS |
| **Backend API** | FastAPI, Uvicorn |
| **Sleep Classifier** | TensorFlow / Keras (CNN-LSTM) |
| **Mood Classifier** | XGBoost, Scikit-Learn |
| **Signal Processing** | MNE-Python, SciPy, NumPy, Pandas |
| **Database & Auth** | SQLite (`sqlite3`), PyJWT, PBKDF2 Password Hashing |
| **Document Export** | ReportLab (PDF) |
| **LLM Narrative** | Anthropic Claude API (via `anthropic` client library) |
| **Testing** | pytest, HTTPX (FastAPI TestClient) |
| **Deployment** | Docker, Docker Compose |

---

## 🏗️ System Architecture

```
                 ┌────────────────────────────────┐
                 │       User (Web Browser)       │
                 └───────────────┬────────────────┘
                                 │ HTTP / WebSockets
                                 ▼ (Port 8501)
                 ┌────────────────────────────────┐
                 │       Streamlit Frontend       │
                 └───────────────┬────────────────┘
                                 │ REST API Calls
                                 ▼ (Port 8000)
                 ┌────────────────────────────────┐
                 │        FastAPI Backend         │
                 └──────────────┬───┬───┬─────────┘
                                │   │   │
        ┌───────────────────────┘   │   └────────────────────────┐
        ▼                           ▼                            ▼
┌────────────────┐          ┌────────────────┐          ┌────────────────┐
│   ML Models    │          │  Data Storage  │          │  External APIs │
├────────────────┤          ├────────────────┤          ├────────────────┤
│ • CNN-LSTM     │          │ • SQLite DB    │          │ • Anthropic    │
│   (Sleep)      │          │ • users.json   │          │   Claude (Opt) │
│ • XGBoost      │          │                │          │                │
│   (Mood)       │          │                │          │                │
└────────────────┘          └────────────────┘          └────────────────┘
```

---

## 📁 Project Structure

Below is the directory layout showing the structure of the repository:

```
An Intelligent System for Sleep Pattern Detection and Dream Mood Prediction/
├── datasets/                            # Dataset directory (Excluded from git)
│   ├── SC4001E0-PSG.edf                 # Sleep-EDF sample PSG
│   ├── SC4001EC-Hypnogram.edf           # Sleep-EDF sample hypnogram
│   └── YAAD ECG+GSR dataset/            # Raw ECG/GSR files
├── models/                              # Pre-trained models and saved metadata
│   ├── sleep_cnn_lstm/
│   │   ├── model.h5                     # CNN-LSTM weights
│   │   └── metrics.json                 # Accuracy and classification reports
│   ├── mood_xgb.model                   # XGBoost classifier binary
│   └── mood_xgb_metrics.json            # XGBoost classification metrics
├── tests/                               # Automated tests folder
│   ├── __init__.py
│   ├── test_api.py                      # FastAPI endpoint tests
│   └── test_extract_features.py         # Test suite for HRV/GSR feature calculations
├── .env.example                         # Environment template config
├── .gitignore                           # Git ignore rules
├── Dockerfile.backend                   # Docker instructions for FastAPI
├── Dockerfile.frontend                  # Docker instructions for Streamlit
├── docker-compose.yml                   # Docker Compose file orchestrating services
├── auth.py                              # Password hashing (PBKDF2) & JWT management
├── database.py                          # SQLite persistence logic
├── dream_mood_module.py                 # Core business logic for mood analysis
├── extract_features.py                  # HRV & GSR signal feature engineering
├── llm_client.py                        # Optional Claude AI prompt and Q&A client
├── main_fastapi.py                      # FastAPI REST application routes
├── pdf_report.py                        # ReportLab PDF report generation utilities
├── preprocess_sleepedf.py               # Preprocessing pipelines for Sleep-EDF files
├── preprocess_yaad.py                   # Preprocessing pipelines for YAAD CSV files
├── README.md                            # Main project documentation
├── requirements.txt                     # Python packages and version locks
├── run.bat                              # Windows runner batch file
├── run_inference.py                     # Execution interface for ML inference runs
├── sample_ecg_test.csv                  # Mock ECG signal data for quick testing
├── sample_gsr_test.csv                  # Mock GSR signal data for quick testing
├── sample_run.sh                        # Bash runner execution script
├── sleep_ai.db                          # SQLite local database file
├── sleep_pattern_analyst.py             # Sleep stage processing & analysis logic
├── streamlit_app.py                     # Streamlit frontend dashboard
├── suggestions.py                       # Rules and logic for recommendations
├── train_mood_model.py                  # Script to train and evaluate XGBoost classifier
└── train_sleep_model.py                 # Script to train and evaluate CNN-LSTM neural net
```

---

## 📊 Data Preprocessing & Feature Engineering

### Sleep Signal Processing (`preprocess_sleepedf.py`)
1. **EDF Parsing**: MNE-Python reads the PSG and Hypnogram European Data Format files.
2. **Channel Selection**: Targets two specific EEG channels (`EEG Fpz-Cz` and `EEG Pz-Oz`).
3. **Filtering**: Applies a FIR bandpass filter between **0.5 Hz and 30 Hz** to remove high-frequency noise and DC drift.
4. **Segmentation**: Cuts continuous signals into **30-second epochs** matching the hypnogram standard (3,000 samples per epoch at 100 Hz).
5. **Normalization**: Conducts Z-score normalization per channel on each individual epoch.
6. **Data Shape**: Final preprocessed array output of size `(N, 2, 3000)`.

### Mood Signal Processing (`preprocess_yaad.py` + `extract_features.py`)
1. **Alignment**: Resamples raw ECG and GSR data to 100 Hz to generate continuous aligned arrays of shape `(N, 2, 3000)`.
2. **HRV Feature Engineering**: Extracts **6 key ECG Heart Rate Variability metrics**:
   - `BPM`: Heart rate (Beats Per Minute)
   - `SDNN`: Standard deviation of NN intervals
   - `RMSSD`: Root mean square of successive differences
   - `pNN50`: Percentage of successive RR intervals differing by >50ms
   - `LF_power`: Low Frequency power band
   - `LF_HF_ratio`: Ratio of Low to High Frequency power
3. **GSR Feature Engineering**: Extracts **5 electrodermal metrics**:
   - `Mean`: Average GSR voltage
   - `Std`: Variance of GSR
   - `Num_Peaks`: Number of skin conductance responses
   - `Mean_Peak_Amplitude`: Average magnitude of GSR peaks
   - `Slope`: Linear drift indicating arousal changes
4. **Feature Combination**: Combines these into a **11-dimensional feature vector** (`features_yaad.csv`).

---

## 🤖 AI Models & Training

### 1. Sleep Staging CNN-LSTM
Designed to capture both local waveform shapes (CNN) and sequential transition dynamics (LSTM).

- **Architecture**:
  ```
  Input Shape: (3000, 2)   [Time Steps, EEG Channels]
      ↓
  Conv1D (64 filters, kernel=3, ReLU activation)
  BatchNormalization + MaxPooling1D (pool_size=2) + Dropout(0.3)
      ↓
  Conv1D (128 filters, kernel=3, ReLU activation)
  BatchNormalization + MaxPooling1D (pool_size=2) + Dropout(0.3)
      ↓
  LSTM (64 units)
  Dropout(0.3)
      ↓
  Dense (32 units, ReLU activation)
  Dense (3 units, Softmax activation) → Output: [Light (0), Deep (1), REM (2)]
  ```
- **Training**: Optimized using **Adam** with **sparse categorical crossentropy** loss. Employs computed class weights to handle dataset imbalances. Metrics are stored inside `models/sleep_cnn_lstm/metrics.json`.

### 2. Dream Mood XGBoost Classifier
Classifies dream mood based on arousal features extracted from physiological markers.

- **Architecture**: Gradient Boosted Decision Tree (via `xgboost`).
- **Features**: 11-dimensional engineered vector (HRV + GSR).
- **Target labels**: `Negative (0)`, `Neutral (1)`, or `Positive (2)`.
- **Metrics**: Evaluated and logged in `models/mood_xgb_metrics.json`.

---

## 🔌 API Endpoints

FastAPI exposes the following endpoints (with optional JWT authorization):

| Endpoint | HTTP Method | Authentication | Description |
| :--- | :--- | :--- | :--- |
| `/status` | `GET` | None | System status (TensorFlow availability, model loading status, Claude status). |
| `/register` | `POST` | None | Register a new user (`username`, `password`, `email`). |
| `/login` | `POST` | None | Log in and receive a JWT Bearer access token. |
| `/history` | `GET` | JWT Token | Retrieve the last 50 session history records for the authorized user. |
| `/predict_sleep_upload` | `POST` | JWT Token | Upload an `.edf` file to perform model-based sleep staging. |
| `/predict_sleep_sample` | `POST` | JWT Token | Perform sleep staging model inference using the default sample PSG. |
| `/predict_mood_upload` | `POST` | JWT Token | Upload ECG+GSR `.csv` data to perform XGBoost mood classification. |
| `/predict_mood_sample` | `POST` | JWT Token | Perform mood classification using sample test values. |
| `/simulate_sleep` | `POST` | JWT Token | Perform a rule-based simulation of sleep metrics. |
| `/simulate_mood` | `POST` | JWT Token | Perform a model-based (or heuristic fallback) mood simulation. |
| `/generate_report` | `POST` | JWT Token | Build a comprehensive structured JSON report for sleep/mood metrics. |
| `/generate_report_pdf` | `POST` | JWT Token | Generate and stream a printable PDF report (ReportLab). |
| `/ask_insight` | `POST` | JWT Token | Grounded chat endpoint to ask Claude questions about current session data. |

---

## 🔧 Getting Started (Local Installation)

### Prerequisites
- Python 3.9–3.12 (highly recommended for TensorFlow and pyEDFlib wheel compatibility).

### Setup Steps

1. **Clone & Navigate**:
   ```bash
   cd "An Intelligent System for Sleep Pattern Detection and Dream Mood Prediction"
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .\.venv\Scripts\Activate.ps1
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project folder (use `.env.example` as a starting template):
   ```ini
   ANTHROPIC_API_KEY=your-api-key-here     # (Optional) For AI narratives & chat Q&A
   JWT_SECRET_KEY=your-random-jwt-secret  # To sign API tokens (Recommended in prod)
   ```

5. **Train Models**:
   Prepare files and train both the deep learning and gradient boosted networks:
   ```bash
   # Preprocess EDF files
   python preprocess_sleepedf.py
   # Preprocess YAAD datasets
   python preprocess_yaad.py
   # Extract ECG+GSR features
   python extract_features.py
   # Train the CNN-LSTM network
   python train_sleep_model.py
   # Train the XGBoost Classifier
   python train_mood_model.py
   ```

6. **Start Server & Dashboard**:
   Run the batch script (Windows only):
   ```cmd
   run.bat
   ```
   Or run manually from the terminal:
   ```bash
   # Start Backend (Port 8000)
   python -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000
   
   # Start Frontend (Port 8501)
   python -m streamlit run streamlit_app.py
   ```

---

## 🐳 Docker Deployment

The application is fully containerized and configured via Docker Compose.

1. **Prepare configuration**:
   Ensure you have a `.env` file containing configuration variables.

2. **Launch via Compose**:
   ```bash
   docker compose up --build
   ```

3. **Endpoints**:
   - Access Streamlit Dashboard at `http://localhost:8501`
   - Access FastAPI Swagger Documentation at `http://localhost:8000/docs`

---

## 🧪 Running Tests

Ensure system integrity by running the test suite:

```bash
# Run tests with verbose output
pytest tests/ -v
```
The test suite validates:
- Signal processing feature extractions (HRV and GSR calculations).
- FastAPI backend router endpoints (mock validation and simulation requests).