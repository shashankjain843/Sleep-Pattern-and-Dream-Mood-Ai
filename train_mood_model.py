# File: train_mood_model.py
import os
import json
import logging
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

METRICS_PATH = "models/mood_xgb_metrics.json"


def train_mood():
    if os.path.exists("models/mood_xgb.model"):
        logger.info("Mood model already exists — skipping training.")
        return

    if not os.path.exists("features_yaad.csv"):
        logger.error("features_yaad.csv not found — run extract_features.py first.")
        return

    df = pd.read_csv("features_yaad.csv")
    X  = df.drop(columns=["label"])
    y  = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_estimators=100,
    )

    logger.info("Training mood XGBoost model…")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    logger.info("Mood model — Test accuracy: %.4f", acc)

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/mood_xgb.model")
    logger.info("Saved model to models/mood_xgb.model")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    class_names = ["Negative", "Neutral", "Positive"]
    report_dict = classification_report(
        y_test, preds, target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "classes": class_names,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_mood()
