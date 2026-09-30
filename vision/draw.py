"""Small drawing helpers shared by collect_data.py and main.py."""

from __future__ import annotations

import cv2
import numpy as np

# Bone pairs follow MediaPipe's landmark ordering (wrist=0 ... pinky tip=20).
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]


def draw_skeleton(image: np.ndarray, landmarks: np.ndarray, mirrored: bool = True) -> None:
    """Draw the 21 landmarks and finger bones onto `image`.

    landmarks: (21, 3) normalized MediaPipe coordinates (unmirrored).
    mirrored:  pass True when `image` was already flipped for display, so
               the x coordinates are flipped back to match it.
    """
    height, width = image.shape[:2]
    points = []
    for x, y, _z in landmarks:
        px = (1.0 - x) if mirrored else x
        points.append((int(px * width), int(y * height)))
    for a, b in BONES:
        cv2.line(image, points[a], points[b], (0, 255, 0), 2)
    for point in points:
        cv2.circle(image, point, 3, (0, 0, 255), -1)
