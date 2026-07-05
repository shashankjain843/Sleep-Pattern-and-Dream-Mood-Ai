# File: train_sleep_model.py
import os
import json
import logging
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

METRICS_PATH = "models/sleep_cnn_lstm/metrics.json"


def train_sleep():
    if os.path.exists("models/sleep_cnn_lstm/model.h5"):
        logger.info("Sleep model already exists — skipping training.")
        return

    if not os.path.exists("X_sleep.npy"):
        logger.error("X_sleep.npy not found — run preprocess_sleepedf.py first.")
        return

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Dense, Conv1D, MaxPooling1D, LSTM, Dropout, BatchNormalization,
        )
        from sklearn.utils import class_weight
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError:
        logger.error("TensorFlow or scikit-learn not installed — cannot train sleep model.")
        return

    X = np.load("X_sleep.npy")
    y = np.load("y_sleep.npy")

    # (N, Channels, Time) → (N, Time, Channels)
    X = np.transpose(X, (0, 2, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))

    # ── Architecture (2 channels, LSTM-64, Dense-32, 5 epochs) ──────────────
    model = Sequential([
        Conv1D(filters=64,  kernel_size=3, activation="relu",
               input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=128, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dense(3,  activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    logger.info("Training sleep CNN-LSTM model…")

    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Sleep model — Test accuracy: %.4f  Loss: %.4f", acc, loss)

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models/sleep_cnn_lstm", exist_ok=True)
    model.save("models/sleep_cnn_lstm/model.h5")
    logger.info("Saved model to models/sleep_cnn_lstm/model.h5")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    class_names = ["Light", "Deep", "REM"]
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "loss": round(float(loss), 4),
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
    train_sleep()
