"""Load the trained model and turn feature vectors into ON / OFF.

The model file is a small joblib bundle created by train_classifier.py:

    {"model": <RandomForestClassifier>, "labels": ["OFF", "ON"],
     "feature_size": 63, "trained_at": "...", "samples": 240}

Wrapping it here means main.py never touches joblib or scikit-learn
directly, and the "model missing / model invalid" error messages live in
one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np

from vision.preprocessing import FEATURE_SIZE


class ModelError(RuntimeError):
    """Raised when the model file is missing or unusable."""


@dataclass
class Prediction:
    label: str          # "ON" or "OFF"
    confidence: float   # 0.0 .. 1.0 (class probability from the model)


class SignClassifier:
    def __init__(self, model, labels: list[str]) -> None:
        self.model = model
        self.labels = labels

    @staticmethod
    def load(path: str) -> "SignClassifier":
        if not os.path.exists(path):
            raise ModelError(
                f"No trained model found at '{path}'.\n"
                "Collect a dataset first, then train:\n"
                "    python recognition/collect_data.py --label ON\n"
                "    python recognition/collect_data.py --label OFF\n"
                "    python recognition/train_classifier.py"
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise ModelError(f"Could not load model '{path}': {exc}") from exc

        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        labels = bundle.get("labels") if isinstance(bundle, dict) else None
        if model is None or not hasattr(model, "predict_proba"):
            raise ModelError(
                f"Model file '{path}' does not contain a usable classifier. "
                "Re-run:  python recognition/train_classifier.py"
            )
        if labels is None:
            labels = list(model.classes_)
        return SignClassifier(model, [str(label) for label in labels])

    def predict(self, features: np.ndarray) -> Prediction:
        """Classify one 63-dim feature vector."""
        vector = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != FEATURE_SIZE:
            raise ModelError(
                f"Feature vector has size {vector.shape[1]}, expected {FEATURE_SIZE}. "
                "Model and preprocessing are out of sync — retrain the model."
            )
        probabilities = self.model.predict_proba(vector)[0]
        best_index = int(np.argmax(probabilities))
        return Prediction(
            label=self.labels[best_index],
            confidence=float(probabilities[best_index]),
        )
