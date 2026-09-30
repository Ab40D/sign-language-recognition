#!/usr/bin/env python3
"""Download the MediaPipe Hand Landmarker model (one time, ~8 MB).

MediaPipe's Tasks API does not bundle the model weights inside the pip
package; they must sit next to the code. This script saves them to:

    models/hand_landmarker.task

Source: Google's official MediaPipe model storage (Apache-2.0 licensed
model weights, same license as MediaPipe itself).

Usage:

    python scripts/fetch_hand_landmarker.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO_ROOT, "models", "hand_landmarker.task")


def main() -> int:
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"Already present: {DEST}")
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading hand_landmarker.task ...")
    tmp = DEST + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / (1 << 20):.1f} / {total / (1 << 20):.1f} MB", end="")
        os.replace(tmp, DEST)
        print(f"\nSaved -> {DEST}")
        return 0
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"\nDownload failed: {exc}")
        print("You can also download it manually from:")
        print("  " + MODEL_URL)
        print(f"and save it as: {DEST}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
