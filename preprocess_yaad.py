# File: preprocess_yaad.py
import os
import logging
import numpy as np
import pandas as pd
from scipy.signal import resample

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR_ECG = "datasets/YAAD ECG+GSR dataset/Raw Data/Multimodal/ECG/"
DATA_DIR_GSR = "datasets/YAAD ECG+GSR dataset/Raw Data/Multimodal/GSR/"
LABEL_DIR    = "datasets/YAAD ECG+GSR dataset/Self-Annotation Labels/"
TARGET_LENGTH = 3000


def load_single_yaad_trial(subject_id, trial_id):
    """
    Load a single YAAD ECG + GSR trial.
    Returns (ecg_array, gsr_array) or None on failure.
    """
    ecg_path = os.path.join(DATA_DIR_ECG, f"{subject_id}_{trial_id}_ECG.csv")
    gsr_path = os.path.join(DATA_DIR_GSR, f"{subject_id}_{trial_id}_GSR.csv")

    if not os.path.exists(ecg_path) or not os.path.exists(gsr_path):
        logger.debug("Trial files not found: %s / %s", ecg_path, gsr_path)
        return None

    try:
        ecg_data = pd.read_csv(ecg_path).values.flatten()
        gsr_data = pd.read_csv(gsr_path).values.flatten()

        ecg_resampled = resample(ecg_data, TARGET_LENGTH)
        gsr_resampled = resample(gsr_data, TARGET_LENGTH)

        return ecg_resampled, gsr_resampled
    except Exception:
        logger.exception("Error loading trial %s/%s", subject_id, trial_id)
        return None


def preprocess_yaad_data():
    X_list, y_list = [], []

    if not os.path.exists(LABEL_DIR):
        logger.warning("YAAD label directory not found (%s). Generating dummy data.", LABEL_DIR)
        X_dummy = np.random.randn(50, 2, 3000).astype(np.float32)
        y_dummy = np.random.randint(0, 3, 50)
        np.save("X_yaad.npy", X_dummy)
        np.save("y_yaad.npy", y_dummy)
        return

    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".csv")]
    logger.info("Found %d label files in %s", len(label_files), LABEL_DIR)

    for l_file in label_files:
        try:
            df_labels = pd.read_csv(os.path.join(LABEL_DIR, l_file))
            subject_id = l_file.split("_")[0]

            for _, row in df_labels.iterrows():
                trial_id = row.get("trial_id", "trial1")
                valence  = row.get("valence", 5)

                if valence > 6:   label = 2
                elif valence < 4: label = 0
                else:             label = 1

                data = load_single_yaad_trial(subject_id, trial_id)
                if data:
                    ecg, gsr = data
                    combined = np.stack([ecg, gsr])
                    X_list.append(combined)
                    y_list.append(label)

        except Exception:
            logger.exception("Error processing label file %s", l_file)
            continue

    if X_list:
        X_yaad = np.array(X_list)
        y_yaad = np.array(y_list)
        np.save("X_yaad.npy", X_yaad)
        np.save("y_yaad.npy", y_yaad)
        logger.info("Saved X_yaad.npy %s, y_yaad.npy %s", X_yaad.shape, y_yaad.shape)
    else:
        logger.warning("No YAAD data processed — saving dummy data")
        X_dummy = np.random.randn(50, 2, 3000).astype(np.float32)
        y_dummy = np.random.randint(0, 3, 50)
        np.save("X_yaad.npy", X_dummy)
        np.save("y_yaad.npy", y_dummy)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    preprocess_yaad_data()
