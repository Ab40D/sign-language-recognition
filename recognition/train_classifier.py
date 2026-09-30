"""Train the ON/OFF sign classifier from collected landmark data.

Pipeline:
  1. Load every .npz session under data/samples/ON and data/samples/OFF.
     (The collector already stored *normalized* landmark vectors via the
     shared vision.preprocessing module, so training does not preprocess
     again — the same numbers the model sees here are the numbers it will
     see at runtime.)
  2. Validate sizes and shapes; fail with a clear message if the dataset
     is missing or too small.
  3. Stratified 80/20 split, train a Random Forest, evaluate, save.

Why Random Forest and not an SVM or a neural net?
  * We have small tabular data (63 features, ~100-300 rows). Random
    Forests are strong in exactly this regime and hard to overfit with
    sensible tree counts.
  * predict_proba gives a confidence score we can threshold (the SVM's
    probabilities are slower and less direct).
  * No feature scaling needed, fast to train, instant to predict —
    ideal for a real-time loop on any laptop.

Usage (from the repository root):

    python recognition/train_classifier.py
    python recognition/train_classifier.py --data-dir data/samples --model-out recognition/model/sign_classifier.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABELS, Config                                # noqa: E402
from vision.feature_extractor import features_have_valid_shape   # noqa: E402
from vision.preprocessing import FEATURE_SIZE                    # noqa: E402

MIN_SAMPLES_PER_CLASS = 10


class DatasetError(RuntimeError):
    pass


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all sessions for both labels. Raises DatasetError on problems."""
    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    if not os.path.isdir(data_dir):
        raise DatasetError(
            f"Dataset directory '{data_dir}' does not exist.\n"
            "Collect samples first:\n"
            "    python recognition/collect_data.py --label ON\n"
            "    python recognition/collect_data.py --label OFF"
        )

    for label in LABELS:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            raise DatasetError(
                f"Missing folder '{label_dir}'. Collect this label first:\n"
                f"    python recognition/collect_data.py --label {label}"
            )
        sessions = sorted(f for f in os.listdir(label_dir) if f.endswith(".npz"))
        if not sessions:
            raise DatasetError(
                f"Folder '{label_dir}' has no .npz sessions. Collect first:\n"
                f"    python recognition/collect_data.py --label {label}"
            )
        count = 0
        for session in sessions:
            path = os.path.join(label_dir, session)
            try:
                with np.load(path) as bundle:
                    features = bundle["features"]
            except Exception as exc:
                raise DatasetError(f"Could not read '{path}': {exc}") from exc
            if features.ndim != 2 or features.shape[1] != FEATURE_SIZE:
                raise DatasetError(
                    f"'{path}' has shape {features.shape}, expected (N, {FEATURE_SIZE}).\n"
                    "Was it produced by this project's collector?"
                )
            if not np.all(np.isfinite(features)):
                raise DatasetError(f"'{path}' contains NaN/inf values — recollect that session.")
            all_features.append(features)
            all_labels.extend([label] * features.shape[0])
            count += features.shape[0]
        print(f"  {label}: {count} samples from {len(sessions)} session(s)")
        if count < MIN_SAMPLES_PER_CLASS:
            raise DatasetError(
                f"Only {count} '{label}' samples — need at least {MIN_SAMPLES_PER_CLASS}.\n"
                f"Collect more:  python recognition/collect_data.py --label {label} --count 60"
            )

    features_matrix = np.vstack(all_features)
    labels_array = np.array(all_labels)
    if not features_have_valid_shape(features_matrix):
        raise DatasetError("Combined dataset is invalid (shape or NaN check failed).")
    return features_matrix, labels_array


def train(data_dir: str, model_path: str, test_size: float, seed: int) -> int:
    print(f"Loading dataset from '{data_dir}'...")
    features, labels = load_dataset(data_dir)
    print(f"Total: {len(labels)} samples, {features.shape[1]} features each.\n")

    # Stratify so train and test keep the same ON/OFF ratio.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    print(f"Train: {len(y_train)}   Test: {len(y_test)}\n")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",  # safe even if ON/OFF counts differ a bit
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    print("=== Evaluation (held-out test set) ===")
    print(f"Accuracy: {accuracy:.3f}\n")
    print(classification_report(y_test, predicted, labels=list(LABELS), zero_division=0))
    matrix = confusion_matrix(y_test, predicted, labels=list(LABELS))
    print(f"Confusion matrix (rows=true, cols=predicted, order={list(LABELS)}):")
    print(matrix, "\n")

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    bundle = {
        "model": model,
        "labels": list(model.classes_),
        "feature_size": FEATURE_SIZE,
        "samples": int(len(labels)),
        "test_accuracy": float(accuracy),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, model_path)
    print(f"Model saved -> {model_path}")
    print("Run the live app now:  python main.py --simulate")
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train the ON/OFF sign classifier.")
    parser.add_argument("--data-dir", default=cfg.data_dir)
    parser.add_argument("--model-out", default=cfg.model_path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        return train(args.data_dir, args.model_out, args.test_size, args.seed)
    except DatasetError as exc:
        print(f"Dataset problem: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
