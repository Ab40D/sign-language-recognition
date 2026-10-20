"""Collect training samples for one sign with your own webcam.
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"


Why self-collected data? Public ASL datasets exist, but none of the
commonly licensed ones contain exactly the two lexicalized signs #ON and
#OFF as *isolated, held poses* suitable for a two-class landmark model.
Collecting 60 samples per sign takes about 30 seconds each and gives the
classifier examples of YOUR hand, YOUR camera, and YOUR lighting — which
is exactly what it will see at runtime.

Each saved sample is the 63-dim normalized landmark vector produced by
vision.preprocessing.normalize_landmarks — the very same function used at
inference time, so collection and recognition can never drift apart.

Usage (from the repository root):

    python recognition/collect_data.py --label ON
    python recognition/collect_data.py --label OFF
    python recognition/collect_data.py --label ON --count 100 --camera 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
import os
import cv2
import numpy as np

# Allow running as `python recognition/collect_data.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABELS, Config                        # noqa: E402
from vision.draw import draw_skeleton                    # noqa: E402
from vision.feature_extractor import extract_features   # noqa: E402
from vision.hand_detector import HandDetector           # noqa: E402

COUNTDOWN_S = 3.0

SIGN_GUIDANCE = {
    "ON": (
        "Sign ON (#ON, lexicalized fingerspelling):",
        "Fingerspell O then N quickly, one hand, and HOLD the N:",
        "index + middle fingers extended over the tucked thumb,",
        "ring + pinky curled into the palm. Palm faces you/left.",
    ),
    "OFF": (
        "Sign OFF (#OFF, lexicalized fingerspelling):",
        "Fingerspell O-F-F quickly, one hand, and HOLD the final F:",
        "thumb tip touches index tip forming a circle,",
        "middle + ring + pinky extended and spread.",
    ),
}


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}.\n"
            "Check that a webcam is connected, close other apps using it,\n"
            "or try another index:  --camera 1"
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def run_collection(label: str, count: int, camera_index: int, interval_s: float) -> int:
    cfg = Config()
    print(f"Collecting {count} '{label}' samples on camera {camera_index}.")
    print("Tips: vary your hand position and distance a little between samples.")
    for line in SIGN_GUIDANCE[label]:
        print("  " + line)

    camera = open_camera(camera_index, cfg.frame_width, cfg.frame_height)
    detector = HandDetector(
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        max_num_hands=2,
    )

    features: list[np.ndarray] = []
    last_capture = 0.0
    window = f"Collect {label} — press Q to abort"

    try:
        # Short countdown so the user can get into position.
        deadline = time.monotonic() + COUNTDOWN_S
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames.")
            remaining = int(deadline - time.monotonic()) + 1
            shown = cv2.flip(frame, 1) if cfg.mirror_view else frame
            cv2.putText(shown, f"Get ready... {remaining}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cfg.color_warn, 2)
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Aborted by user.")
                return 1

        while len(features) < count:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped delivering frames; keeping what we have.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = detector.detect(rgb)
            shown = cv2.flip(frame, 1) if cfg.mirror_view else frame

            status = ""
            now = time.monotonic()
            if len(hands) == 0:
                status = "No hand detected — show exactly one hand"
            elif len(hands) > 1:
                status = "Two hands visible — show only ONE hand"
            elif now - last_capture >= interval_s:
                vector = extract_features(hands[0])
                features.append(vector)
                last_capture = now
                status = f"Captured {len(features)}/{count}"
            else:
                status = f"Capturing... {len(features)}/{count}"

            # Draw landmarks of the first hand for feedback.
            if hands and cfg.show_landmarks:
                for hand in hands:
                    draw_skeleton(shown, hand.landmarks, mirrored=cfg.mirror_view)

            progress = f"{label}: {len(features)}/{count}"
            cv2.putText(shown, progress, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, cfg.color_ok, 2)
            cv2.putText(shown, status, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.color_text, 2)
            cv2.imshow(window, shown)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Aborted by user.")
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    if len(features) < max(5, count // 4):
        print(f"Only {len(features)} samples captured — not saving. Try again.")
        return 1

    out_dir = os.path.join(cfg.data_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_dir, f"session_{stamp}.npz")
    matrix = np.vstack(features)
    np.savez_compressed(out_path, features=matrix, labels=np.array([label] * len(features)))
    print(f"Saved {len(features)} samples -> {out_path}")
    print("Next: collect the other label, then run  python recognition/train_classifier.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect landmark samples for one ASL sign.")
    parser.add_argument("--label", required=True, choices=LABELS,
                        help="Which sign to collect: ON or OFF")
    parser.add_argument("--count", type=int, default=Config().collect_count,
                        help="Number of samples to capture (default 60)")
    parser.add_argument("--camera", type=int, default=Config().camera_index,
                        help="Camera index (default 0)")
    parser.add_argument("--interval", type=float, default=Config().collect_interval_s,
                        help="Seconds between captures (default 0.10)")
    args = parser.parse_args(argv)
    try:
        return run_collection(args.label, args.count, args.camera, args.interval)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
