# File: preprocess_sleepedf.py
import os
import logging
import numpy as np
import mne
from mne.io import read_raw_edf
from mne import read_annotations

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "datasets"
PSG_FILES = ["SC4001E0-PSG.edf", "SC4002E0-PSG.edf"]
HYP_FILES = ["SC4001EC-Hypnogram.edf", "SC4002EC-Hypnogram.edf"]
SAMPLING_RATE = 100
WINDOW_SIZE = 30
# Actual channels used (2, not 4 — matches real preprocessing code)
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
LABEL_MAP = {
    "Sleep stage W": 0,  # Light/Wake
    "Sleep stage 1": 0,  # Light
    "Sleep stage 2": 0,  # Light
    "Sleep stage 3": 1,  # Deep
    "Sleep stage 4": 1,  # Deep
    "Sleep stage R": 2,  # REM
}


def load_and_preprocess_file(psg_path, hyp_path=None):
    """
    Load a single EDF file and return epochs + labels (if hypnogram provided).
    Returns (X, y) where X.shape == (N, 2, 3000) and y.shape == (N,).
    Both are None on failure.
    """
    if not os.path.exists(psg_path):
        logger.warning("PSG file not found: %s", psg_path)
        return None, None

    try:
        raw = read_raw_edf(psg_path, preload=True, verbose=False)

        # Select channels — fall back to first 2 if exact names not present
        available_chans = raw.ch_names
        selected_chans = [ch for ch in CHANNELS if ch in available_chans]
        if not selected_chans:
            logger.warning(
                "Expected channels %s not found in %s — using first 2 available",
                CHANNELS, psg_path,
            )
            selected_chans = available_chans[:2]

        raw.pick(selected_chans)
        raw.filter(0.5, 30.0, fir_design="firwin", verbose=False)
        logger.info("Loaded %s — channels: %s", psg_path, selected_chans)

        if hyp_path and os.path.exists(hyp_path):
            annot = read_annotations(hyp_path)
            raw.set_annotations(annot, emit_warning=False)
            events, event_id = mne.events_from_annotations(raw, event_id=None, verbose=False)
        else:
            logger.info("No hypnogram provided — using fixed-length segmentation")
            events = None
            event_id = None

        tmax = 30.0 - 1.0 / raw.info["sfreq"]

        if events is not None:
            epochs = mne.Epochs(
                raw, events, event_id=event_id, tmin=0.0, tmax=tmax,
                baseline=None, verbose=False,
            )

            X, y = [], []
            for stage_name, stage_id in event_id.items():
                if stage_name not in LABEL_MAP:
                    continue
                target_class = LABEL_MAP[stage_name]
                stage_epochs = epochs[stage_name]
                data = stage_epochs.get_data()
                if len(data) > 0:
                    data = (data - np.mean(data, axis=2, keepdims=True)) / (
                        np.std(data, axis=2, keepdims=True) + 1e-8
                    )
                    X.append(data)
                    y.append(np.full(len(data), target_class))

            if X:
                X_out = np.concatenate(X, axis=0)
                y_out = np.concatenate(y, axis=0)
                logger.info(
                    "Preprocessed %s — X: %s  y: %s", psg_path, X_out.shape, y_out.shape
                )
                return X_out, y_out
            logger.warning("No labelled epochs extracted from %s", psg_path)
            return None, None

        # No hypnogram — plain segmentation
        epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True, verbose=False)
        data = epochs.get_data()
        data = (data - np.mean(data, axis=2, keepdims=True)) / (
            np.std(data, axis=2, keepdims=True) + 1e-8
        )
        logger.info("Fixed-length segmentation — X: %s", data.shape)
        return data, None

    except Exception:
        logger.exception("Error processing %s", psg_path)
        return None, None


def preprocess_sleep_data():
    X_all, y_all = [], []

    for psg_file, hyp_file in zip(PSG_FILES, HYP_FILES):
        psg_path = os.path.join(DATA_DIR, psg_file)
        hyp_path = os.path.join(DATA_DIR, hyp_file)

        X, y = load_and_preprocess_file(psg_path, hyp_path)
        if X is not None and y is not None:
            X_all.append(X)
            y_all.append(y)

    if X_all:
        X_sleep = np.concatenate(X_all, axis=0)
        y_sleep = np.concatenate(y_all, axis=0)
        np.save("X_sleep.npy", X_sleep)
        np.save("y_sleep.npy", y_sleep)
        logger.info("Saved X_sleep.npy %s, y_sleep.npy %s", X_sleep.shape, y_sleep.shape)
    else:
        logger.warning("No data processed — generating dummy sleep data")
        X_dummy = np.random.randn(100, 2, 3000).astype(np.float32)
        y_dummy = np.random.randint(0, 3, 100)
        np.save("X_sleep.npy", X_dummy)
        np.save("y_sleep.npy", y_dummy)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    preprocess_sleep_data()
