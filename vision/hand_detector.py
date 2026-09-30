"""Hand detection with MediaPipe Hand Landmarker (Tasks API).

How MediaPipe detects the hand (short version):
  Hand Landmarker is a two-stage model. Stage 1 finds the palm (a square
  region) in the image with a lightweight detector. Stage 2 crops that
  region and regresses 21 3D hand landmarks (x, y, z) plus handedness.
  We only use the landmarks; the raw pixels never reach our classifier.

The model weights are NOT inside the mediapipe pip package. They live in
models/hand_landmarker.task and are fetched once with:

    python scripts/fetch_hand_landmarker.py

This module wraps MediaPipe so the rest of the project never imports
mediapipe directly. That keeps the pipeline testable: anything that needs
"a hand detector" can receive a fake one with the same interface.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

# Quieten MediaPipe's C++ logging (must happen before mediapipe is imported).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

NUM_LANDMARKS = 21

# Default location of the MediaPipe model, relative to the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "hand_landmarker.task")

MODEL_HELP = (
    "MediaPipe model file not found.\n"
    "Download it once with:\n"
    "    python scripts/fetch_hand_landmarker.py\n"
    "(it saves models/hand_landmarker.task, ~8 MB)"
)


@dataclass
class HandDetection:
    """One detected hand, reduced to what the pipeline needs.

    landmarks:  numpy array of shape (21, 3). x and y are normalized to
                [0, 1] relative to the image; z is relative depth with the
                wrist as origin (smaller = closer to the camera).
    handedness: "Right" or "Left" as reported by MediaPipe. NOTE: on a
                normal (unmirrored) webcam image MediaPipe labels as if
                looking at the hand, so a user's right hand usually
                arrives as "Left". We only use this to mirror landmarks
                into one canonical orientation (see preprocessing.py).
    """

    landmarks: np.ndarray
    handedness: str


class HandDetector:
    """Wrapper around mediapipe.tasks HandLandmarker in VIDEO mode.

    VIDEO mode uses the previous frame for tracking, which is what we want
    for a webcam loop. It requires strictly increasing timestamps, so we
    feed it milliseconds from a monotonic clock.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(MODEL_HELP)

        # Imported lazily so unit tests can run without mediapipe installed.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,   # 2 so we can NOTICE when 2 hands appear
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp_image = mp.Image                  # convenience aliases
        self._mp_image_format = mp.ImageFormat
        self._start_ms = time.monotonic() * 1000.0
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray) -> list[HandDetection]:
        """Run detection on an RGB frame and return all hands found.

        frame_rgb: numpy array (H, W, 3), RGB order. MediaPipe expects RGB;
        OpenCV gives BGR, so the caller converts first.
        """
        mp_image = self._mp_image(
            image_format=self._mp_image_format.SRGB, data=np.ascontiguousarray(frame_rgb)
        )
        # VIDEO mode needs strictly increasing timestamps in milliseconds.
        timestamp_ms = int(time.monotonic() * 1000.0 - self._start_ms)
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections: list[HandDetection] = []
        for index, hand_landmarks in enumerate(result.hand_landmarks):
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64
            )
            handedness = "Right"
            if index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name or "Right"
            detections.append(HandDetection(landmarks=points, handedness=handedness))
        return detections

    def close(self) -> None:
        self._landmarker.close()
