# File: extract_features.py
import os
import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

logger = logging.getLogger(__name__)


def compute_hrv(ecg_signal, fs: int = 100) -> list[float]:
    """
    Compute 6 HRV features from an ECG signal.
    Returns [bpm, sdnn, rmssd, pnn50, lf, lf_hf].
    Returns [0]*6 if the signal has fewer than 2 peaks.
    """
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.5)
    if len(peaks) < 2:
        return [0.0] * 6

    rr_intervals = np.diff(peaks) / fs * 1000  # ms

    bpm   = 60_000 / np.mean(rr_intervals)
    sdnn  = float(np.std(rr_intervals))
    rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr_intervals))))) if len(rr_intervals) > 1 else 0.0
    pnn50 = float(
        np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
    ) if len(rr_intervals) > 1 else 0.0

    f, Pxx = welch(rr_intervals, fs=4.0, nperseg=min(len(rr_intervals), 256))
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    lf  = float(integrate(Pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]))
    hf  = float(integrate(Pxx[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)]))
    lf_hf = lf / hf if hf > 0 else 0.0

    return [float(bpm), sdnn, rmssd, pnn50, lf, lf_hf]


def compute_gsr(gsr_signal, fs: int = 100) -> list[float]:
    """
    Compute 5 GSR features.
    Returns [mean_gsr, std_gsr, num_peaks, mean_peak_amp, slope].
    """
    mean_gsr = float(np.mean(gsr_signal))
    std_gsr  = float(np.std(gsr_signal))

    peaks, properties = find_peaks(gsr_signal, prominence=0.05)
    num_peaks      = len(peaks)
    mean_peak_amp  = float(np.mean(properties["prominences"])) if num_peaks > 0 else 0.0

    slope = float(np.polyfit(np.arange(len(gsr_signal)), gsr_signal, 1)[0])

    return [mean_gsr, std_gsr, float(num_peaks), mean_peak_amp, slope]


def extract_features():
    try:
        if not os.path.exists("X_sleep.npy"):
            logger.warning("X_sleep.npy not found — skipping sleep feature extraction")
        else:
            X_sleep = np.load("X_sleep.npy")
            feats_sleep = []
            for i in range(len(X_sleep)):
                sig = X_sleep[i, 0, :]
                feats_sleep.append([np.mean(sig), np.std(sig), 0, 0, 0, 0])

            df_sleep = pd.DataFrame(feats_sleep, columns=["mean", "std", "f3", "f4", "f5", "f6"])
            df_sleep.to_csv("features_sleep.csv", index=False)
            logger.info("Saved features_sleep.csv (%d rows)", len(df_sleep))

        if not os.path.exists("X_yaad.npy"):
            logger.warning("X_yaad.npy not found — skipping mood feature extraction")
            return

        X_yaad = np.load("X_yaad.npy")
        y_yaad = np.load("y_yaad.npy")

        feats_yaad = []
        for i in range(len(X_yaad)):
            ecg = X_yaad[i, 0, :]
            gsr = X_yaad[i, 1, :]
            feats_yaad.append(compute_hrv(ecg) + compute_gsr(gsr))

        cols = [
            "bpm", "sdnn", "rmssd", "pnn50", "lf", "lf_hf",
            "mean_gsr", "std_gsr", "num_peaks", "mean_peak_amp", "slope",
        ]
        df_yaad = pd.DataFrame(feats_yaad, columns=cols)
        df_yaad["label"] = y_yaad
        df_yaad.to_csv("features_yaad.csv", index=False)
        logger.info("Saved features_yaad.csv (%d rows)", len(df_yaad))

    except Exception:
        logger.exception("Error during feature extraction")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    extract_features()
