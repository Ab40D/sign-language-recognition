from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vision.hand_detector import HandDetector
from vision.preprocessing import normalize_landmarks


def extract_video(video_path, label, detector, sample_every):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    samples = []
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = detector.detect(rgb)

        if hands:
            landmarks = hands[0].landmarks
            features = normalize_landmarks(landmarks)
            samples.append(features)

        frame_index += 1

    cap.release()

    if not samples:
        raise RuntimeError(f"No hands detected in {video_path}")

    print(
        f"{video_path}: detected {len(samples)} samples "
        f"from {frame_index} frames"
    )

    return np.asarray(samples, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--output-dir", default="data/samples")
    parser.add_argument("--sample-every", type=int, default=3)

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    detector = HandDetector()

    try:
        persons = sorted(
            p for p in video_dir.iterdir()
            if p.is_dir()
        )

        if not persons:
            raise RuntimeError(
                f"No person folders found in {video_dir}"
            )

        for person_dir in persons:

            print(f"\n=== {person_dir.name} ===")

            for label in ("ON", "OFF"):

                video_path = person_dir / f"{label}.mp4"

                if not video_path.exists():
                    print(f"Missing: {video_path}")
                    continue

                features = extract_video(
                    video_path,
                    label,
                    detector,
                    args.sample_every,
                )

                label_dir = output_dir / label
                label_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                output_path = (
                    label_dir
                    / f"{person_dir.name}_{label}.npz"
                )

                np.savez_compressed(
                    output_path,
                    features=features,
                    labels=np.array(
                        [label] * len(features)
                    ),
                )

                print(
                    f"Saved {len(features)} samples -> "
                    f"{output_path}"
                )

    finally:
        detector.close()


if __name__ == "__main__":
    main()
