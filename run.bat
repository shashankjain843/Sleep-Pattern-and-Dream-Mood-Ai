@echo off
setlocal

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

set "PYTHON=python"
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
)

if exist requirements.txt (
    "%PYTHON%" -m pip install -r requirements.txt
) else (
    echo requirements.txt not found
)

if exist preprocess_sleepedf.py "%PYTHON%" preprocess_sleepedf.py
if exist preprocess_yaad.py "%PYTHON%" preprocess_yaad.py
if exist extract_features.py "%PYTHON%" extract_features.py
if exist train_sleep_model.py "%PYTHON%" train_sleep_model.py
if exist train_mood_model.py "%PYTHON%" train_mood_model.py
if exist run_inference.py "%PYTHON%" run_inference.py

if exist main_fastapi.py (
    start "FastAPI Backend" "%PYTHON%" -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000
)

if exist streamlit_app.py (
    "%PYTHON%" -m streamlit run streamlit_app.py
)

pause
