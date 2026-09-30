"""Turn a HandDetection into the classifier's feature vector.

This is the only place that combines handedness + landmark normalization,
so data collection and real-time inference cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from vision.hand_detector import HandDetection
from vision.preprocessing import FEATURE_SIZE, normalize_landmarks


def extract_features(detection: HandDetection) -> np.ndarray:
    """Return the normalized 63-dim feature vector for one hand."""
    return normalize_landmarks(detection.landmarks, detection.handedness)


def features_have_valid_shape(features: np.ndarray) -> bool:
    """Cheap sanity check used when loading datasets and models."""
    return (
        isinstance(features, np.ndarray)
        and features.ndim == 2
        and features.shape[1] == FEATURE_SIZE
        and np.all(np.isfinite(features))
    )
