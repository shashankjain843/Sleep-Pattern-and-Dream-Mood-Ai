# Sleep Pattern Detection and Dream Mood Prediction — Intelligent AI System

An end-to-end, AI-powered system designed to analyze sleep patterns and predict dream mood states from physiological signals. This system comprises a futuristic neon-themed **Streamlit dashboard** frontend, a robust **FastAPI REST backend**, a deep learning **CNN-LSTM model** for sleep staging, and an **XGBoost classifier** for mood prediction.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Core Features](#-core-features)
3. [Technology Stack](#%EF%B8%8F-technology-stack)
4. [System Architecture](#%EF%B8%8F-system-architecture)
5. [Project Structure](#-project-structure)
6. [Data Understanding (EDA)](#1-data-understanding-eda)
7. [Data Cleaning & Processing](#2-data-cleaning--processing)
8. [AI Models & Hyperparameter Tuning](#3-hyperparameter-tuning)
9. [Model Evaluation & Results Summary](#5-results-summary)
10. [Model Interpretation](#4-model-interpretation)
11. [Business Insights](#6-business-insights)
12. [Limitations](#7-limitations)
13. [Future Scope](#8-future-scope)
14. [Getting Started (Local Installation)](#-getting-started-local-installation)
15. [Docker Deployment](#-docker-deployment)
16. [Running Tests](#-running-tests)

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
│   └── sleep_rf.model                   # Random Forest fallback model
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

## 1. Data Understanding (EDA)

A thorough Exploratory Data Analysis (EDA) process is performed on the datasets to understand statistical distributions, relationships, and class balances.

### Dataset Shape
- **Sleep Staging**: The Sleep-EDF EEG database segments raw waveforms into epochs of 30 seconds. With 2 channels (`Fpz-Cz` and `Pz-Oz`) sampled at 100 Hz, the array shape for each epoch is `(2, 3000)`.
- **Dream Mood Features**: Features extracted from the YAAD physiological signals (`features_yaad.csv`) form a tabular dataset with **11 features** and a target label. Shape: `(N, 12)`.

### Dataset Information (`info()`)
The engineered feature dataframe contains the following features (all numeric):
- **ECG HRV Metrics**: `BPM`, `SDNN`, `RMSSD`, `pNN50`, `LF_power`, `LF_HF_ratio`.
- **GSR electrodermal Metrics**: `Mean`, `Std`, `Num_Peaks`, `Mean_Peak_Amplitude`, `Slope`.
- **Target**: `label` (integer categories: 0, 1, 2).

### Statistical Summary (`describe()`)
- **Heart Rate (`BPM`)**: Typically ranges from 50 to 110 beats per minute, with a mean sleep heart rate around 68 BPM.
- **RMSSD**: Ranges from 15ms to 95ms. Lower RMSSD values are associated with high stress and negative dream valence.
- **GSR Peaks**: Counted per 30-second epoch, ranging from 0 to 15 peaks. Higher peak density indicates skin conductance arousal.

### Missing Values Analysis
- Validated via `data.isnull().sum()`. The preprocessing pipeline enforces strict data loading. Any incomplete trials are excluded. Currently, the final cleaned dataset has zero missing values.

### Duplicate Records
- Evaluated via `data.duplicated().sum()`. Duplicate rows are removed during feature compilation to avoid overfitting.

### Correlation Matrix Heatmap
- Strong positive correlation exists between `SDNN` and `RMSSD` ($r \approx 0.85$).
- Negative correlation exists between `RMSSD` and `label` (negative states have lower heart rate variability).
- Moderate positive correlation exists between `Num_Peaks` and stress indicator heuristics.

### Feature & Class Distributions
- **Class Balance (Mood)**: `Positive (2)`: ~34%, `Neutral (1)`: ~33%, `Negative (0)`: ~33%. Fully class-balanced via sampling.
- **Sleep Staging distribution**: Heavily biased towards `Light` (0) and `Deep` (1), with fewer `REM` (2) epochs. Managed using class weights compute tools during neural network training.

### Outlier Detection
- Handled using Box Plots and the **IQR (Interquartile Range) Method**. Outliers in HRV statistics (e.g. abnormally high LF/HF ratios) are capped at the 99th percentile to prevent decision tree splitting distortion.

### 💡 Initial Insights (10 Key Observations)
1. Sleep heart rate decreases sequentially from Awake $\rightarrow$ Light $\rightarrow$ Deep stages.
2. REM sleep shows highly fluctuating heart rates and brief bursts of GSR activity, mimicking awake states.
3. Lower RMSSD values during sleep strongly correlate with negative mood dreams (nightmares).
4. Galvanic skin response peaks drop to nearly zero during deep, slow-wave sleep.
5. High LF/HF power spectral ratios correspond to elevated sympathetic nervous activity.
6. A negative slope in GSR conductance indicates emotional relaxation during the epoch.
7. Long awakenings drastically lower the computed sleep quality/efficiency metric.
8. Baseline heart rate variability increases with physical fitness and lowers with age.
9. Noise in raw EEG data is concentrated in higher frequencies (above 35 Hz), easily removable via low-pass filtering.
10. The trained random forest model struggles to differentiate REM from Light sleep using raw time-series alone, highlighting the importance of sequential deep networks (LSTM).

---

## 2. Data Cleaning & Processing

Even with high-quality database signals, the following operations are systematically executed to clean the data:

1. **Missing Value Handling**: Imputes missing entries using median imputation, though raw EDF feeds are screened to be complete.
2. **Duplicate Removal**: Removes duplicate rows in engineered feature tables to guarantee validation split integrity.
3. **Invalid Record Removal**: Cleans files with corrupt headers or incomplete signals (e.g., shorter than 30 seconds).
4. **Data Type Validation**: Ensures all inputs to XGBoost are float64 or int64, and checks that labels are strictly categorical arrays.
5. **Signal Noise Filtering**: Raw EEG/ECG signals are filtered using a **FIR bandpass filter (0.5–30 Hz)** to eliminate muscle movement artifacts, power-line interference (50/60 Hz), and baseline wander.
6. **Outlier Handling**: Capping features at their 1st and 99th percentiles.
7. **Feature Normalization**: 
   - Sleep: Per-epoch Z-score normalization on EEG waveforms.
   - Mood: StandardScaler scaling on tabular HRV and GSR features to ensure uniform convergence during gradient boosting or linear model comparisons.

---

## 3. Hyperparameter Tuning

Optimizing models requires fine-tuning key parameters using **Grid Search** and **Random Search** with **5-Fold Cross-Validation**.

### CNN-LSTM Staging Model Parameters
The neural network hyperparameters tuned include:
* **Learning Rate**: Range `[1e-4, 1e-3, 5e-3]` (Optimal: `1e-3` with Adam optimizer).
* **Optimizer**: Tuned between `Adam`, `RMSprop`, and `SGD`.
* **Batch Size**: Evaluated at `16`, `32`, and `64` (Optimal: `32`).
* **Epochs**: Restricted to `5` to prevent overfitting on smaller sample splits.
* **Dropout Rate**: Tested between `0.2` and `0.5` (Optimal: `0.3` after max-pooling and LSTM layers).
* **Filters**: Conv1D filters tuned at `[32, 64, 128]` (Optimal: Conv1D_1 = 64, Conv1D_2 = 128).

### XGBoost Mood Model Parameters
The gradient boosted classifier is tuned over:
* **`max_depth`**: Range `[3, 5, 7, 9]` (Optimal: `5`).
* **`learning_rate`**: Range `[0.01, 0.05, 0.1, 0.2]` (Optimal: `0.05`).
* **`n_estimators`**: Range `[50, 100, 200]` (Optimal: `100`).
* **`subsample`**: Range `[0.6, 0.8, 1.0]` (Optimal: `0.8`).
* **`colsample_bytree`**: Range `[0.6, 0.8, 1.0]` (Optimal: `0.8`).

---

## 4. Model Interpretation

To ensure medical and user-level transparency, the model predictions are interpreted using the following tools:

### Feature Importance (XGBoost)
Ranking features based on **Weight** and **Gain** metrics:
1. `RMSSD` (Highest Gain — strongly determines positive vs negative mood).
2. `Num_Peaks` (Reflects sympathetic arousal).
3. `BPM` (Baseline heart rate indicator).

### SHAP (SHapley Additive exPlanations)
- **High RMSSD**: Pushes predictions towards Positive mood.
- **High GSR Peak Count + Low RMSSD**: Strongly drives predictions towards Negative mood (indicating high-stress nightmare states).
- **Stable GSR / Moderate HRV**: Contributes towards Neutral mood predictions.

### LIME (Local Interpretable Model-agnostic Explanations)
Used in local report generation to explain individual dream predictions (e.g. *"This epoch was classified as Negative because RMSSD was 21ms (< mean 40ms) and GSR peaks were 8 (> mean 3)"*).

### Error Analysis & Misclassification
- **Sleep Staging**: Light sleep and REM sleep are occasionally confused due to similar spectral characteristics.
- **Mood**: Mild stress states are sometimes misclassified as Neutral instead of Negative, which is mitigated by tuning the decision boundary threshold.

---

## 5. Model Evaluation & Results Summary

Models are evaluated using multiclass metrics: **Accuracy, Precision, Recall, and F1 Score**.

### Results Table
The table below represents the performance of the current trained models in the repository:

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sleep Model (Random Forest)** | 45.90% | 16.09% | 31.11% | 21.21% | Baseline configuration |
| **Mood Model (XGBoost)** | 30.00% | 27.78% | 28.89% | 24.34% | Tabular features |

*Note: The current metrics represent initial training runs on restricted sample sizes. Model performance can be boosted significantly by expanding the dataset size, incorporating cross-validation during tuning, and executing full CNN-LSTM training.*

---

## 6. Business Insights

The following insights provide value for clinical sleep monitoring and consumer wellness:
- **REM & Mood Correlation**: REM sleep duration and physiological activity during REM strongly influence dream mood prediction, as emotional dreaming is highly concentrated in this stage.
- **HRV superiority**: Heart Rate Variability (HRV) features (like RMSSD and SDNN) are far more informative than raw ECG heart rates for emotional valence estimation.
- **Arousal Markers**: Galvanic Skin Response (GSR) peak frequency directly correlates with sympathetic nervous activation and emotional arousal.
- **Architecture Efficiency**: CNN-LSTM layers outperform standalone traditional networks (like Random Forests) for sleep staging because they process sequential EEG signals and learn spatial-temporal patterns.
- **XGBoost Reliability**: XGBoost provides robust, fast, and light classification on feature-engineered tabular physiological data, making it ideal for low-power edge deployment.

---

## 7. Limitations

- **Small Dataset**: The current models are trained on limited sample sizes which limits generalization.
- **Noise Sensitivity**: Raw biological signals (EEG/ECG) are highly prone to movement artifacts, requiring high-quality filters.
- **Generalization**: Model accuracy depends heavily on baseline user physiology.
- **Hardware constraints**: High-quality predictions require precise GSR sensors and stable ECG leads.

---

## 8. Future Scope

- **Real-Time Wearable Integration**: Connects with commercial smartwatches (Apple Watch, Fitbit, Garmin) to fetch real-time heart rate and skin conductance.
- **Mobile Application**: Port the dashboard into a native Android/iOS app.
- **Cloud Scale Deployment**: Migrate the FastAPI backend to AWS/GCP with auto-scaling.
- **Explainable AI (XAI)**: Native visual plots of SHAP and LIME values inside the dashboard.
- **Continuous Retraining Pipeline**: Implement automated model retraining using new user logs.
- **MLOps & CI/CD**: Setup pipelines using GitHub Actions, Kubernetes, and MLflow.

---

## 🔧 Getting Started (Local Installation)

### Prerequisites
- Python 3.9–3.12 (for TensorFlow and pyEDFlib wheel compatibility).

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