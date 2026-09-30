"""End-to-end test without any hardware:

    synthetic landmarks -> normalization -> train Random Forest
    -> load model -> simulated frame loop -> SignStabilizer
    -> command mapper -> SimulatedSerialController

It verifies the exact chain the live app uses, minus the webcam pixels.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.command_parser import sign_to_command              # noqa: E402
from control.debounce import SignStabilizer                     # noqa: E402
from control.serial_controller import SimulatedSerialController  # noqa: E402
from recognition.classifier import SignClassifier               # noqa: E402
from recognition.train_classifier import train                  # noqa: E402
from tests.synthetic_hands import make_dataset, make_sample     # noqa: E402
from vision.preprocessing import normalize_landmarks            # noqa: E402

FRAME = 1.0 / 30.0


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Build a small synthetic dataset on disk, shaped exactly like the
        # real collector's output (data/samples/<LABEL>/session_*.npz).
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.data_dir = os.path.join(cls.tmpdir.name, "samples")
        cls.model_path = os.path.join(cls.tmpdir.name, "model.pkl")

        features, labels = make_dataset(n_per_class=120, seed=11)
        for label in ("ON", "OFF"):
            label_dir = os.path.join(cls.data_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            mask = np.array(labels) == label
            np.savez_compressed(
                os.path.join(label_dir, "session_test.npz"),
                features=features[mask],
                labels=np.array(labels)[mask],
            )
        # Train through the real training script entry point.
        exit_code = train(cls.data_dir, cls.model_path, test_size=0.25, seed=3)
        assert exit_code == 0, "training failed on the synthetic dataset"
        cls.classifier = SignClassifier.load(cls.model_path)

    def test_classifier_learns_both_signs(self) -> None:
        rng = np.random.default_rng(99)
        correct = 0
        total = 60
        for _ in range(total):
            for label in ("ON", "OFF"):
                features = normalize_landmarks(make_sample(label, rng), "Right")
                prediction = self.classifier.predict(features)
                if prediction.label == label and prediction.confidence > 0.6:
                    correct += 1
        accuracy = correct / total
        self.assertGreaterEqual(accuracy, 0.95, f"accuracy too low: {accuracy}")

    def test_full_control_chain_sends_each_command_once(self) -> None:
        """The core acceptance test:

        hold ON  -> LIGHT_ON sent exactly once
        keep ON  -> nothing more
        hold OFF -> LIGHT_OFF sent exactly once
        hand away, then ON again -> LIGHT_ON sent again
        """
        rng = np.random.default_rng(5)
        stabilizer = SignStabilizer(stable_frames=6, min_hold_seconds=0.3,
                                    cooldown_seconds=0.5, none_reset_frames=4)
        serial = SimulatedSerialController()
        serial.connect()

        def recognize(label: str | None) -> str | None:
            """Simulate one frame of the vision stage."""
            if label is None:
                return None
            features = normalize_landmarks(make_sample(label, rng), "Right")
            prediction = self.classifier.predict(features)
            return prediction.label if prediction.confidence >= 0.7 else None

        frames = ["ON"] * 90 + ["OFF"] * 90 + [None] * 20 + ["ON"] * 90
        now = 0.0
        for raw_label in frames:
            sign = recognize(raw_label)
            fired = stabilizer.update(sign, now)
            if fired is not None:
                command = sign_to_command(fired)
                assert command is not None
                serial.send_command(command)
            now += FRAME

        self.assertEqual(
            serial.commands_received, ["LIGHT_ON", "LIGHT_OFF", "LIGHT_ON"]
        )
        self.assertEqual(serial.last_response, "LIGHT_STATE:ON")


if __name__ == "__main__":
    unittest.main()
