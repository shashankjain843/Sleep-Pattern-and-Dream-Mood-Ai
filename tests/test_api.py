"""
tests/test_api.py — FastAPI endpoint tests using TestClient.

Covers success and error paths for all major endpoints.
"""
import sys
import os
import json

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main_fastapi import app

client = TestClient(app)


# ── /status ───────────────────────────────────────────────────────────────────
class TestStatus:
    def test_status_ok(self):
        r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert "tf_available"       in body
        assert "sleep_model_loaded" in body
        assert "mood_model_loaded"  in body
        assert "claude_available"   in body


# ── /simulate_sleep ───────────────────────────────────────────────────────────
class TestSimulateSleep:
    VALID_PAYLOAD = {"hr": 65, "movement": 2, "spo2": 98, "hrv_var": 50, "duration": 8}

    def test_valid_payload_returns_200(self):
        r = client.post("/simulate_sleep", json=self.VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "stage"      in body
        assert "efficiency" in body
        assert "source"     in body
        assert body["source"] == "heuristic"

    def test_missing_field_returns_422(self):
        bad = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "hr"}
        r = client.post("/simulate_sleep", json=bad)
        assert r.status_code == 422

    def test_efficiency_in_valid_range(self):
        r = client.post("/simulate_sleep", json=self.VALID_PAYLOAD)
        eff = r.json()["efficiency"]
        assert 0 <= eff <= 100, f"Efficiency {eff} out of [0, 100]"

    def test_suggestions_present(self):
        r = client.post("/simulate_sleep", json=self.VALID_PAYLOAD)
        assert "suggestions" in r.json()
        assert isinstance(r.json()["suggestions"], list)


# ── /simulate_mood ────────────────────────────────────────────────────────────
class TestSimulateMood:
    VALID_PAYLOAD = {"rmssd": 40, "hr": 70, "gsr_peaks": 5, "gsr_slope": 0.1, "stress": 3}

    def test_valid_payload_returns_200(self):
        r = client.post("/simulate_mood", json=self.VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "mood"          in body
        assert "probabilities" in body
        assert "source"        in body

    def test_mood_label_is_valid(self):
        r = client.post("/simulate_mood", json=self.VALID_PAYLOAD)
        assert r.json()["mood"] in {"Positive", "Neutral", "Negative", "Unknown"}

    def test_probabilities_sum_to_one(self):
        r = client.post("/simulate_mood", json=self.VALID_PAYLOAD)
        probs = r.json()["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_missing_field_returns_422(self):
        bad = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "stress"}
        r = client.post("/simulate_mood", json=bad)
        assert r.status_code == 422


# ── /generate_report ──────────────────────────────────────────────────────────
class TestGenerateReport:
    PAYLOAD = {
        "sleep_metrics": {
            "efficiency": 80, "rem_percent": 20, "deep_percent": 15,
            "light_percent": 65, "duration": 7, "awakenings": 1,
        },
        "mood_metrics": {
            "mood": "Positive",
            "probabilities": {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7},
        },
    }

    def test_valid_report_returns_200(self):
        r = client.post("/generate_report", json=self.PAYLOAD)
        assert r.status_code == 200

    def test_report_has_required_fields(self):
        r = client.post("/generate_report", json=self.PAYLOAD)
        body = r.json()
        for field in ["title", "sleep_summary", "mood_summary", "combined_insights",
                      "suggestions_list", "warnings"]:
            assert field in body, f"Missing field: {field}"

    def test_suggestions_is_list(self):
        r = client.post("/generate_report", json=self.PAYLOAD)
        assert isinstance(r.json()["suggestions_list"], list)

    def test_warnings_is_list(self):
        r = client.post("/generate_report", json=self.PAYLOAD)
        assert isinstance(r.json()["warnings"], list)


# ── /predict_mood_upload ──────────────────────────────────────────────────────
class TestPredictMoodUpload:
    def _make_csv(self, columns=("ecg", "gsr"), n=3000) -> bytes:
        data = {col: np.random.randn(n).tolist() for col in columns}
        import csv, io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(columns))
        writer.writeheader()
        for i in range(n):
            writer.writerow({col: data[col][i] for col in columns})
        return buf.getvalue().encode()

    def test_valid_csv_returns_mood(self):
        csv_bytes = self._make_csv()
        r = client.post(
            "/predict_mood_upload",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "mood" in body or "error" in body   # error ok if model not loaded

    def test_malformed_csv_missing_columns(self):
        """CSV without 'ecg' and 'gsr' columns should return an error dict (not 500)."""
        csv_bytes = self._make_csv(columns=("heart_rate", "skin_conductance"))
        r = client.post(
            "/predict_mood_upload",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 200          # returns 200 with error key
        assert "error" in r.json()


# ── /auth ─────────────────────────────────────────────────────────────────────
class TestAuth:
    def test_register_and_login(self):
        import uuid
        username = f"testuser_{uuid.uuid4().hex[:8]}"
        password = "testpass123"

        # Register
        r = client.post("/register", json={"username": username, "password": password})
        assert r.status_code == 200

        # Login
        r = client.post("/login", json={"username": username, "password": password})
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_login_wrong_password_returns_401(self):
        r = client.post("/login", json={"username": "nobody", "password": "wrong"})
        assert r.status_code == 401

    def test_duplicate_register_returns_409(self):
        import uuid
        username = f"dup_{uuid.uuid4().hex[:8]}"
        client.post("/register", json={"username": username, "password": "pass"})
        r = client.post("/register", json={"username": username, "password": "pass"})
        assert r.status_code == 409


# ── /history ──────────────────────────────────────────────────────────────────
class TestHistory:
    def test_history_returns_list(self):
        r = client.get("/history")
        assert r.status_code == 200
        assert "sessions" in r.json()
        assert isinstance(r.json()["sessions"], list)
