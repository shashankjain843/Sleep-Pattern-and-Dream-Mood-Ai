# File: sample_run.sh
#!/bin/bash

# Detect Python
if command -v py &> /dev/null; then
    PYTHON=py
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Using Python: $PYTHON"

# 0. Install Requirements
echo "Installing requirements..."
$PYTHON -m pip install -r requirements.txt

# 1. Preprocess
echo "Running Preprocessing..."
$PYTHON preprocess_sleepedf.py
$PYTHON preprocess_yaad.py

# 2. Extract Features
echo "Extracting Features..."
$PYTHON extract_features.py

# 3. Train Models
echo "Training Models..."
$PYTHON train_sleep_model.py
$PYTHON train_mood_model.py

# 4. Run Inference Test
echo "Running Inference Test..."
$PYTHON run_inference.py

# 5. Start Backend (Background)
echo "Starting FastAPI..."
$PYTHON -m uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 &

# 6. Start Frontend
echo "Starting Streamlit..."
$PYTHON -m streamlit run streamlit_app.py
