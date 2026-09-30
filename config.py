"""Central configuration for the ASL device controller.

Every tunable setting lives here so the rest of the code never hardcodes
magic numbers. main.py builds a Config from these defaults and lets the
command line override individual fields.

If you change a default here, it applies everywhere (collection,
training, and real-time inference all read from this module).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The two signs this prototype recognizes. Labels are used everywhere:
# dataset folders, classifier output, and the command map.
LABEL_ON = "ON"
LABEL_OFF = "OFF"
LABELS = (LABEL_ON, LABEL_OFF)

# Serial protocol tokens understood by the Arduino sketch.
# Keep in sync with arduino/sign_controller/sign_controller.ino.
PROTOCOL_COMMANDS = ("LIGHT_ON", "LIGHT_OFF", "STATUS")


@dataclass
class Config:
    # --- Webcam ---------------------------------------------------------
    camera_index: int = 0              # OpenCV camera index (0 = default)
    frame_width: int = 640             # requested capture size
    frame_height: int = 480

    # --- MediaPipe hand detection ----------------------------------------
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2             # used to *detect* "more than one hand"

    # --- Recognition ------------------------------------------------------
    model_path: str = "recognition/model/sign_classifier.pkl"
    confidence_threshold: float = 0.70   # below this -> UNKNOWN, nothing sent

    # --- Stability / debounce ---------------------------------------------
    stable_frames: int = 6             # consecutive frames of the same sign
    min_hold_seconds: float = 0.5      # and held at least this long
    cooldown_seconds: float = 1.0      # ignore triggers this long after a send
    none_reset_frames: int = 4         # hand absent this many frames -> reset state

    # --- Arduino serial ----------------------------------------------------
    serial_port: str | None = None     # None = auto-detect; e.g. /dev/ttyACM0, COM3
    baud_rate: int = 9600
    ready_timeout_s: float = 6.0       # wait for READY after the board resets
    response_timeout_s: float = 2.0    # wait for a reply after each command
    reconnect_interval_s: float = 3.0  # retry a dropped port every N seconds

    # --- Dataset collection -------------------------------------------------
    data_dir: str = "data/samples"
    collect_count: int = 60            # samples per collection run
    collect_interval_s: float = 0.10   # ~10 samples/second while collecting

    # --- Display -------------------------------------------------------------
    mirror_view: bool = True           # mirror webcam so it feels like a mirror
    show_landmarks: bool = True
    font_scale: float = 0.6
    # UI colors (BGR)
    color_ok: tuple = (0, 200, 0)
    color_warn: tuple = (0, 165, 255)
    color_bad: tuple = (80, 80, 230)
    color_text: tuple = (255, 255, 255)
    color_panel: tuple = (40, 40, 40)

    # Keyboard test mode is always active in the live window:
    #   O = inject ON, F = inject OFF, C = clear state, Q = quit
