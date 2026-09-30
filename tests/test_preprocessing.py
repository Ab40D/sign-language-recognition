"""Tests for vision.preprocessing: the normalization that collection,
training, and inference all share."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.preprocessing import FEATURE_SIZE, normalize_landmarks  # noqa: E402
from tests.synthetic_hands import make_sample  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_output_shape_and_finite(self) -> None:
        raw = make_sample("ON", self.rng)
        features = normalize_landmarks(raw, "Right")
        self.assertEqual(features.shape, (FEATURE_SIZE,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_translation_invariance(self) -> None:
        raw = make_sample("ON", self.rng)
        moved = raw + np.array([0.23, -0.31, 0.0])
        np.testing.assert_allclose(
            normalize_landmarks(raw, "Right"),
            normalize_landmarks(moved, "Right"),
            atol=1e-9,
        )

    def test_scale_invariance(self) -> None:
        raw = make_sample("OFF", self.rng)
        scaled = raw * 2.7
        np.testing.assert_allclose(
            normalize_landmarks(raw, "Right"),
            normalize_landmarks(scaled, "Right"),
            atol=1e-9,
        )

    def test_rotation_invariance(self) -> None:
        raw = make_sample("ON", self.rng)
        angle = np.deg2rad(33)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = raw.copy()
        rotated[:, :2] = rotated[:, :2] @ rot.T
        np.testing.assert_allclose(
            normalize_landmarks(raw, "Right"),
            normalize_landmarks(rotated, "Right"),
            atol=1e-9,
        )

    def test_handedness_mirroring_is_canonicalized(self) -> None:
        """A 'Left' hand must be mirrored into the same canonical space."""
        raw_right = make_sample("OFF", self.rng)
        raw_left = raw_right.copy()
        raw_left[:, 0] = -raw_left[:, 0] + 2 * raw_left[:, 0].mean()
        np.testing.assert_allclose(
            normalize_landmarks(raw_right, "Right"),
            normalize_landmarks(raw_left, "Left"),
            atol=1e-9,
        )

    def test_different_shapes_give_different_features(self) -> None:
        on = normalize_landmarks(make_sample("ON", self.rng), "Right")
        off = normalize_landmarks(make_sample("OFF", self.rng), "Right")
        self.assertGreater(float(np.linalg.norm(on - off)), 0.1)

    def test_bad_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_landmarks(np.zeros((5, 3)), "Right")


if __name__ == "__main__":
    unittest.main()
