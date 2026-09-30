"""Synthetic 21-point hand landmark generator for headless testing.

These are NOT real MediaPipe outputs. They are structurally plausible
landmark sets shaped like the two holds this prototype uses:

  ON  -> N-like hold (index + middle extended over the tucked thumb)
  OFF -> F hold (thumb-index circle, other three fingers spread)

They exist so the numeric pipeline (preprocessing, training, stability,
serial simulation) can be verified end-to-end on a machine without a
webcam. Real training data always comes from collect_data.py.
"""

from __future__ import annotations

import numpy as np

# Canonical right-hand skeleton, wrist at origin, fingers pointing "up"
# (-y, like MediaPipe image coordinates). Rough proportions only.
_BASE = np.array([
    [0.00, 0.00, 0.00],    # 0 wrist
    [-0.35, -0.15, 0.00],  # 1 thumb CMC
    [-0.45, -0.30, 0.00],  # 2 thumb MCP
    [-0.52, -0.42, 0.00],  # 3 thumb IP
    [-0.58, -0.52, 0.00],  # 4 thumb TIP
    [-0.15, -0.45, 0.00],  # 5 index MCP
    [-0.17, -0.62, 0.00],  # 6 index PIP
    [-0.18, -0.74, 0.00],  # 7 index DIP
    [-0.19, -0.85, 0.00],  # 8 index TIP
    [-0.02, -0.47, 0.00],  # 9 middle MCP
    [-0.02, -0.66, 0.00],  # 10 middle PIP
    [-0.02, -0.79, 0.00],  # 11 middle DIP
    [-0.02, -0.90, 0.00],  # 12 middle TIP
    [0.10, -0.45, 0.00],   # 13 ring MCP
    [0.11, -0.63, 0.00],   # 14 ring PIP
    [0.12, -0.75, 0.00],   # 15 ring DIP
    [0.12, -0.86, 0.00],   # 16 ring TIP
    [0.22, -0.42, 0.00],   # 17 pinky MCP
    [0.24, -0.56, 0.00],   # 18 pinky PIP
    [0.25, -0.66, 0.00],   # 19 pinky DIP
    [0.26, -0.75, 0.00],   # 20 pinky TIP
], dtype=np.float64)


def _n_shape() -> np.ndarray:
    """Hold of #ON: index + middle extended; ring/pinky curled; thumb tucks
    across the palm behind them."""
    points = _BASE.copy()
    # Curl ring finger into the palm.
    points[14] = [0.10, -0.40, -0.04]
    points[15] = [0.09, -0.33, -0.07]
    points[16] = [0.08, -0.28, -0.09]
    # Curl pinky into the palm.
    points[18] = [0.20, -0.38, -0.04]
    points[19] = [0.18, -0.31, -0.07]
    points[20] = [0.16, -0.26, -0.09]
    # Thumb crosses the palm in front of the curled fingers.
    points[2] = [-0.30, -0.25, -0.05]
    points[3] = [-0.12, -0.30, -0.10]
    points[4] = [0.02, -0.32, -0.13]
    return points


def _f_shape() -> np.ndarray:
    """Hold of #OFF: thumb tip meets index tip (a circle); middle, ring and
    pinky stay extended and spread."""
    points = _BASE.copy()
    ring_center = np.array([-0.30, -0.42, -0.10])
    # Index bends down into the thumb ring.
    points[6] = [-0.22, -0.55, -0.03]
    points[7] = [-0.26, -0.47, -0.08]
    points[8] = ring_center + [0.03, 0.02, 0.00]
    # Thumb reaches up to meet the index tip.
    points[3] = [-0.38, -0.36, -0.06]
    points[4] = ring_center
    # Spread the three free fingers a little wider.
    points[9:13, 0] -= 0.02
    points[13:17, 0] += 0.03
    points[17:21, 0] += 0.06
    return points


SHAPES = {"ON": _n_shape, "OFF": _f_shape}


def make_sample(label: str, rng: np.random.Generator) -> np.ndarray:
    """One jittered (21, 3) landmark array, in 'image-like' coordinates:
    everything positive, roughly in [0, 1], like raw MediaPipe output."""
    points = SHAPES[label]()

    # Per-landmark noise (handshape stays clearly recognizable).
    points += rng.normal(0.0, 0.008, size=points.shape)

    # Random in-plane rotation around the wrist.
    angle = rng.uniform(-0.7, 0.7)  # +-40 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    points[:, :2] = points[:, :2] @ rotation.T

    # Random scale (distance from camera) and z compression.
    scale = rng.uniform(0.45, 1.1)
    points *= scale
    points[:, 2] *= 0.5

    # Random position in frame (wrist lands somewhere reasonable).
    points[:, 0] += rng.uniform(0.30, 0.65)
    points[:, 1] += rng.uniform(0.70, 0.95)

    return points


def make_dataset(n_per_class: int, seed: int = 7) -> tuple[np.ndarray, list[str]]:
    """Build a labeled dataset of synthetic landmarks."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label in ("ON", "OFF"):
        for _ in range(n_per_class):
            from vision.preprocessing import normalize_landmarks

            raw = make_sample(label, rng)
            # Mirror-handedness variant half the time: flip x and label Left,
            # normalization must undo that and land in the same space.
            if rng.random() < 0.5:
                raw = raw.copy()
                raw[:, 0] = -raw[:, 0] + 2 * raw[:, 0].mean()
                features.append(normalize_landmarks(raw, handedness="Left"))
            else:
                features.append(normalize_landmarks(raw, handedness="Right"))
            labels.append(label)
    return np.vstack(features), labels
