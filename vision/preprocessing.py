"""Landmark normalization: ONE shared pipeline for collection, training,
and real-time inference.

Raw MediaPipe landmarks depend on where the hand is in the frame, how big
it is, how it is tilted, and which hand is used. A classifier trained on
raw coordinates would memorize position instead of handshape. So before
any landmark set becomes a feature vector, it goes through these steps:

  1. MIRROR HANDEDNESS  if MediaPipe says "Left", flip x so both hands
                        look like one canonical hand. (Webcam images are
                        not mirrored internally, so the user's right hand
                        is usually reported as "Left".)
  2. TRANSLATE          subtract the wrist (landmark 0) so the hand is
                        anchored at the origin. Moving the hand left/right
                        or up/down no longer changes the features.
  3. SCALE              divide by the largest distance from the wrist to
                        any landmark. Moving closer or farther from the
                        camera no longer changes the features.
  4. ROTATE (in-plane)  rotate around the camera axis so the wrist ->
                        middle-finger-MCP direction points straight "up"
                        (-y). Tilting the wrist no longer changes the
                        features much. z is left untouched.
  5. FLATTEN            produce a 63-dimensional vector (21 points x 3).

Every module in the project must call normalize_landmarks() — never
re-implement any of these steps elsewhere.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
FEATURE_SIZE = NUM_LANDMARKS * 3  # 63

WRIST = 0
MIDDLE_FINGER_MCP = 9  # landmark index used as the "up" reference direction


def normalize_landmarks(
    landmarks: np.ndarray, handedness: str = "Right"
) -> np.ndarray:
    """Normalize a (21, 3) landmark array into a (63,) feature vector.

    landmarks: array of shape (21, 3) straight from MediaPipe.
    handedness: the MediaPipe label ("Right"/"Left"); used to mirror the
                hand so both hands produce comparable features.
    Returns a float64 vector of length 63.
    """
    points = np.asarray(landmarks, dtype=np.float64).copy()
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"Expected landmarks of shape ({NUM_LANDMARKS}, 3), got {points.shape}"
        )

    # Step 1: canonical handedness. Mirror x around the wrist when the
    # detection is a "Left" hand, so left/right hands look the same.
    if handedness == "Left":
        points[:, 0] = -points[:, 0]

    # Step 2: translation invariance — wrist becomes the origin.
    points -= points[WRIST]

    # Step 3: scale invariance — normalize by the longest wrist->landmark
    # distance. Guard against a degenerate (zero-size) detection.
    distances = np.linalg.norm(points, axis=1)
    scale = float(distances.max())
    if scale < 1e-9:
        return np.zeros(FEATURE_SIZE, dtype=np.float64)
    points /= scale

    # Step 4: in-plane rotation so wrist->middle-MCP points along -y.
    reference = points[MIDDLE_FINGER_MCP, :2]  # (x, y) of middle finger MCP
    ref_length = float(np.linalg.norm(reference))
    if ref_length > 1e-9:
        # Angle of the reference vector; we want it at -90 degrees (up).
        angle = np.arctan2(reference[1], reference[0])
        rotation = -angle - np.pi / 2
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points[:, :2] = points[:, :2] @ rot_matrix.T

    # Step 5: flatten to the final feature vector.
    return points.reshape(-1)
