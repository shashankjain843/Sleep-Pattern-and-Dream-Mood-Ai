# File: train_mood_model.py
import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

METRICS_PATH = "models/mood_xgb_metrics.json"


def train_mood(force_train: bool = True):
    if not force_train and os.path.exists("models/mood_xgb.model") and os.path.exists("models/mood_scaler.joblib"):
        logger.info("Mood model and scaler already exist — skipping training.")
        return

    if not os.path.exists("features_yaad.csv"):
        logger.error("features_yaad.csv not found — run extract_features.py first.")
        return

    df = pd.read_csv("features_yaad.csv")
    X  = df.drop(columns=["label"])
    y  = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Feature Normalization (StandardScaler) ──────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/mood_scaler.joblib")
    logger.info("Saved scaler to models/mood_scaler.joblib")

    # ── Baseline Model ────────────────────────────────────────────────────────
    baseline_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_estimators=100,
        random_state=42
    )
    baseline_model.fit(X_train_scaled, y_train)
    baseline_preds = baseline_model.predict(X_test_scaled)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    logger.info("Baseline Mood Model — Test accuracy: %.4f", baseline_acc)

    # ── Hyperparameter Tuning (GridSearchCV) ───────────────────────────────────
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [50, 100]
    }
    logger.info("Running GridSearchCV hyperparameter tuning…")
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    logger.info("GridSearchCV complete. Best Params: %s, Best CV Score: %.4f", best_params, best_score)

    # ── Tuned Model Evaluation ────────────────────────────────────────────────
    preds = best_model.predict(X_test_scaled)
    acc   = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    logger.info("Tuned Mood Model — Test accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", acc, prec, rec, f1)

    # ── Feature Importance Extraction ──────────────────────────────────────────
    importances = best_model.feature_importances_.tolist()
    feature_names = list(X.columns)
    feat_importance_dict = dict(zip(feature_names, importances))
    
    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(best_model, "models/mood_xgb.model")
    logger.info("Saved tuned model to models/mood_xgb.model")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    class_names = ["Negative", "Neutral", "Positive"]
    report_dict = classification_report(
        y_test, preds, target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "baseline_accuracy": round(float(baseline_acc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "best_params": best_params,
        "feature_importances": feat_importance_dict,
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
