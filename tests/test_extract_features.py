"""
tests/test_extract_features.py — Unit tests for compute_hrv and compute_gsr.

Uses synthetic signals with known properties to catch regressions:
  - A sine wave at 1 Hz simulates R-peaks that produce ~60 BPM
  - A signal with known prominence peaks for GSR tests
"""
import sys
import os
import math
import numpy as np
import pytest

# Add the project root to sys.path so we can import directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_features import compute_hrv, compute_gsr


# ── HRV tests ─────────────────────────────────────────────────────────────────
class TestComputeHRV:

    def test_sine_wave_gives_approx_60bpm(self):
        """
        A 1 Hz sine wave at 100 Hz sampling has peaks every 100 samples (1 s intervals),
        so the expected BPM ≈ 60.
        """
        fs = 100
        t  = np.linspace(0, 10, fs * 10, endpoint=False)
        ecg = np.sin(2 * math.pi * 1.0 * t)   # 1 Hz → 1 peak/s → 60 BPM

        result = compute_hrv(ecg, fs=fs)

        assert len(result) == 6, "compute_hrv must return exactly 6 features"
        bpm = result[0]
        assert 50 < bpm < 75, f"Expected BPM ≈ 60, got {bpm:.1f}"

    def test_flat_signal_returns_zeros(self):
        """A flat signal has no peaks → should return [0]*6 without crashing."""
        flat_signal = np.zeros(3000)
        result = compute_hrv(flat_signal)
        assert result == [0.0] * 6

    def test_single_peak_returns_zeros(self):
        """Only one peak → can't compute RR intervals → should return [0]*6."""
        signal = np.zeros(3000)
        signal[500] = 1.0           # single spike
        result = compute_hrv(signal)
        assert result == [0.0] * 6

    def test_returns_non_negative_values(self):
        """All HRV features should be non-negative for a valid signal."""
        fs = 100
        t  = np.linspace(0, 30, fs * 30, endpoint=False)
        ecg = np.sin(2 * math.pi * 1.2 * t)   # ~72 BPM
        result = compute_hrv(ecg, fs=fs)
        assert all(v >= 0 for v in result), f"Got negative features: {result}"

    def test_higher_frequency_gives_higher_bpm(self):
        """2 Hz sine should produce ~120 BPM — sanity-check scaling."""
        fs = 100
        t  = np.linspace(0, 10, fs * 10, endpoint=False)
        ecg_60  = np.sin(2 * math.pi * 1.0 * t)
        ecg_120 = np.sin(2 * math.pi * 2.0 * t)
        bpm_60  = compute_hrv(ecg_60,  fs=fs)[0]
        bpm_120 = compute_hrv(ecg_120, fs=fs)[0]
        assert bpm_120 > bpm_60, "Higher frequency signal should produce higher BPM"


# ── GSR tests ─────────────────────────────────────────────────────────────────
class TestComputeGSR:

    def test_known_peaks_detected(self):
        """
        A flat baseline with 3 sharp spikes should return num_peaks == 3.
        """
        gsr = np.zeros(3000)
        for idx in [500, 1000, 2000]:
            gsr[idx]     = 1.0
            gsr[idx + 1] = 0.5
        result = compute_gsr(gsr)
        assert len(result) == 5, "compute_gsr must return exactly 5 features"
        num_peaks = result[2]
        assert num_peaks >= 3, f"Expected ≥3 peaks, got {num_peaks}"

    def test_flat_signal_zero_peaks(self):
        """A completely flat signal should have 0 peaks."""
        flat = np.ones(3000) * 0.5
        result = compute_gsr(flat)
        num_peaks = result[2]
        assert num_peaks == 0, f"Expected 0 peaks on flat signal, got {num_peaks}"

    def test_flat_signal_near_zero_slope(self):
        """A flat signal should have a slope very close to 0."""
        flat = np.ones(3000) * 2.0
        result = compute_gsr(flat)
        slope = result[4]
        assert abs(slope) < 1e-6, f"Expected slope ≈ 0, got {slope}"

    def test_increasing_signal_positive_slope(self):
        """A linearly increasing signal should have a positive slope."""
        increasing = np.linspace(0, 10, 3000)
        result = compute_gsr(increasing)
        slope = result[4]
        assert slope > 0, f"Expected positive slope, got {slope}"

    def test_mean_std_correct(self):
        """Mean and std should match numpy for a known signal."""
        gsr = np.random.default_rng(42).normal(loc=5.0, scale=1.5, size=3000)
        result = compute_gsr(gsr)
        assert abs(result[0] - np.mean(gsr)) < 1e-6, "Mean mismatch"
        assert abs(result[1] - np.std(gsr))  < 1e-6, "Std mismatch"
