# Dataset directory

This directory holds the training data **you collect with your own webcam**.

Nothing here ships with the repository: signs are personal (your hand, your
camera, your lighting), and no suitable *licensed* public dataset exists for
the two specific held signs this prototype uses (#ON and #OFF — see the main
README, "ASL Reference").

## Layout

```text
data/
└── samples/
    ├── ON/
    │   └── session_2026-09-30_14-12-00.npz
    └── OFF/
        └── session_2026-09-30_14-15-30.npz
```

Each `.npz` session contains:

* `features` — an `(N, 63)` float matrix. Every row is one hand pose,
  normalized by `vision/preprocessing.normalize_landmarks()` at capture
  time (the same function inference uses).
* `labels`   — the matching label strings ("ON" or "OFF").

## Collecting

```bash
python recognition/collect_data.py --label ON
python recognition/collect_data.py --label OFF
```

Roughly 60 samples per label is a good start (~30 seconds each). Vary your
hand position and distance a little between captures. Run the collector
again any time to add more sessions — training merges them all.

## After collecting

```bash
python recognition/train_classifier.py
```

## Cleaning up

Delete any `session_*.npz` file to remove a bad recording, then retrain.
