#!/usr/bin/env python3
"""ASL Device Controller — real-time application.

Pipeline:

    WEBCAM -> OpenCV -> MediaPipe hand landmarks
           -> normalization -> Random Forest classifier -> ON / OFF
           -> stability check -> command mapper -> USB serial -> Arduino -> LEDs

Everything heavy lives in small modules; this file only wires them
together and draws the UI.

Usage:

    python main.py                    # real Arduino, auto-detected port
    python main.py --simulate         # no hardware needed
    python main.py --port /dev/ttyACM0
    python main.py --port COM3        # Windows
    python main.py --send LIGHT_ON    # one-shot test, no model needed
    python main.py --list-ports

Keyboard (development/testing only):
    O = inject ON (held until cleared)   F = inject OFF
    C = clear recognition state          Q = quit
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from config import LABEL_OFF, LABEL_ON, Config
from control.command_parser import is_valid_command, sign_to_command
from control.debounce import SignStabilizer
from control.serial_controller import (
    SerialController,
    SimulatedSerialController,
    choose_port,
    list_serial_ports,
)
from recognition.classifier import ModelError, SignClassifier
from vision.draw import draw_skeleton
from vision.feature_extractor import extract_features
from vision.hand_detector import HandDetector

WINDOW_NAME = "ASL Device Controller"


# --------------------------------------------------------------------------
# One-shot command mode (--send) — lets you test the serial link and the
# Arduino without any model or webcam. Reused concept from the voice project.
# --------------------------------------------------------------------------

def run_one_shot(command: str, cfg: Config, simulate: bool) -> int:
    token = command.strip().upper()
    if not is_valid_command(token):
        print(f"Refusing to send {command!r}. Valid tokens: LIGHT_ON, LIGHT_OFF, STATUS")
        return 2
    controller = _build_controller(cfg, simulate)
    if not controller.connect():
        print("Could not connect to the Arduino.")
        return 1
    try:
        print(f"Sending {token} ...")
        response = controller.send_command(token)
        print(f"Arduino: {response or '(no response — timeout)'}")
        return 0 if response.startswith("LIGHT_STATE:") else 1
    finally:
        controller.close()


def _build_controller(cfg: Config, simulate: bool):
    if simulate:
        print("Arduino: SIMULATION mode (no hardware)")
        return SimulatedSerialController()
    port = choose_port(cfg.serial_port)
    print(f"Port: {port}   Baud: {cfg.baud_rate}")
    controller = SerialController(
        port=port,
        baud=cfg.baud_rate,
        ready_timeout=cfg.ready_timeout_s,
        response_timeout=cfg.response_timeout_s,
        reconnect_interval=cfg.reconnect_interval_s,
    )
    print("Waiting for Arduino READY...")
    controller.connect()
    return controller


# --------------------------------------------------------------------------
# UI drawing
# --------------------------------------------------------------------------

def draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple]], font_scale: float) -> None:
    """Draw a dark panel with status lines at the top of the frame."""
    line_height = int(28 * font_scale / 0.6)
    panel_height = line_height * len(lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    y = line_height + 4
    for text, color in lines:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 2)
        y += line_height


def draw_hand_status(frame: np.ndarray, text: str, color: tuple, font_scale: float) -> None:
    height = frame.shape[0]
    cv2.putText(frame, text, (16, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2)


# --------------------------------------------------------------------------
# Main real-time loop
# --------------------------------------------------------------------------

def run_live(cfg: Config, simulate: bool) -> int:
    # 1. Model first: fail early with a helpful message if it is missing.
    try:
        classifier = SignClassifier.load(cfg.model_path)
    except ModelError as exc:
        print(f"Model problem: {exc}")
        return 1
    print(f"Model loaded from {cfg.model_path}")

    # 2. Arduino link (real or simulated).
    try:
        controller = _build_controller(cfg, simulate)
    except RuntimeError as exc:
        print(f"Serial problem: {exc}")
        print("Tip: test without hardware using  python main.py --simulate")
        return 1

    # 3. Webcam.
    camera = cv2.VideoCapture(cfg.camera_index)
    if not camera.isOpened():
        print(f"Webcam unavailable: could not open camera index {cfg.camera_index}.")
        print("Connect a webcam, close other apps using it, or try --camera 1.")
        controller.close()
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    # 4. Hand detection + stability state machine.
    detector = HandDetector(
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        max_num_hands=cfg.max_num_hands,
    )
    stabilizer = SignStabilizer(
        stable_frames=cfg.stable_frames,
        min_hold_seconds=cfg.min_hold_seconds,
        cooldown_seconds=cfg.cooldown_seconds,
        none_reset_frames=cfg.none_reset_frames,
    )

    last_command = "-"
    last_response = "-"
    detected_sign = "NONE"
    confidence = 0.0
    keyboard_sign: str | None = None  # persistent injection for O/F keys
    camera_failures = 0

    print("Running. Keys: O=inject ON  F=inject OFF  C=clear  Q=quit")
    try:
        while True:
            controller.maybe_reconnect()

            ok, frame = camera.read()
            if not ok:
                camera_failures += 1
                if camera_failures > 30:
                    print("Webcam stopped delivering frames. Exiting.")
                    break
                continue
            camera_failures = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = detector.detect(rgb)

            # --- Recognition ------------------------------------------------
            now = time.monotonic()
            if keyboard_sign is not None:
                # Development injection: bypasses vision, still goes through
                # the stabilizer so cooldown/debounce behave exactly as live.
                detected_sign = keyboard_sign
                confidence = 1.0
                sign_for_stabilizer: str | None = keyboard_sign
            elif len(hands) == 0:
                detected_sign = "NONE"
                confidence = 0.0
                sign_for_stabilizer = None
            elif len(hands) > 1:
                detected_sign = "NONE"
                confidence = 0.0
                sign_for_stabilizer = None
            else:
                features = extract_features(hands[0])
                prediction = classifier.predict(features)
                detected_sign = prediction.label
                confidence = prediction.confidence
                if confidence < cfg.confidence_threshold:
                    detected_sign = "UNKNOWN"
                    sign_for_stabilizer = None
                else:
                    sign_for_stabilizer = prediction.label

            # --- Stability + command mapping + serial ------------------------
            fired = stabilizer.update(sign_for_stabilizer, now)
            if fired is not None:
                command = sign_to_command(fired)
                response = controller.send_command(command) if command else ""
                last_command = command or "-"
                last_response = response or "(timeout)"
                print(f"{fired} stable -> sent {last_command} -> {last_response}")

            # --- Display ------------------------------------------------------
            shown = cv2.flip(frame, 1) if cfg.mirror_view else frame
            if cfg.show_landmarks and hands and keyboard_sign is None:
                for hand in hands:
                    draw_skeleton(shown, hand.landmarks, mirrored=cfg.mirror_view)

            arduino_text = controller.status_text()
            arduino_color = (cfg.color_ok if arduino_text in ("CONNECTED", "SIMULATION")
                             else cfg.color_bad)
            sign_color = cfg.color_ok if detected_sign in (LABEL_ON, LABEL_OFF) else (
                cfg.color_warn if detected_sign == "UNKNOWN" else cfg.color_text)

            lines = [
                ("ASL DEVICE CONTROLLER", cfg.color_text),
                (f"Detected Sign: {detected_sign}   Confidence: {int(confidence * 100)}%",
                 sign_color),
                (f"Command: {last_command}", cfg.color_text),
                (f"Arduino: {arduino_text}", arduino_color),
                (f"Last Response: {last_response}", cfg.color_text),
            ]
            draw_panel(shown, lines, cfg.font_scale)

            if len(hands) > 1:
                draw_hand_status(shown, "Show exactly ONE hand", cfg.color_warn, cfg.font_scale)
            elif len(hands) == 0 and keyboard_sign is None:
                draw_hand_status(shown, "No hand detected", cfg.color_text, cfg.font_scale)
            if keyboard_sign is not None:
                draw_hand_status(shown, f"KEYBOARD TEST: holding {keyboard_sign} (C to clear)",
                                 cfg.color_warn, cfg.font_scale)

            cv2.imshow(WINDOW_NAME, shown)

            # --- Keyboard ------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                keyboard_sign = LABEL_ON
                stabilizer.reset()
            elif key == ord("f"):
                keyboard_sign = LABEL_OFF
                stabilizer.reset()
            elif key == ord("c"):
                keyboard_sign = None
                stabilizer.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        camera.release()
        detector.close()
        controller.close()
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="ASL-based hand-sign recognition prototype for device control."
    )
    parser.add_argument("--port", help="Serial port, e.g. /dev/ttyACM0 or COM3 "
                                       "(default: auto-detect)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run without an Arduino (simulated responses)")
    parser.add_argument("--camera", type=int, default=cfg.camera_index,
                        help="Camera index (default 0)")
    parser.add_argument("--confidence", type=float, default=cfg.confidence_threshold,
                        help="Minimum classifier confidence to accept a sign")
    parser.add_argument("--stable-frames", type=int, default=cfg.stable_frames,
                        help="Consecutive frames needed before a sign triggers")
    parser.add_argument("--cooldown", type=float, default=cfg.cooldown_seconds,
                        help="Seconds to ignore triggers after sending a command")
    parser.add_argument("--send", metavar="COMMAND",
                        help="Send one token (LIGHT_ON/LIGHT_OFF/STATUS) and exit")
    parser.add_argument("--list-ports", action="store_true",
                        help="List serial ports and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.port is not None:
        cfg.serial_port = args.port
    if args.camera is not None:
        cfg.camera_index = args.camera
    cfg.confidence_threshold = args.confidence
    cfg.stable_frames = args.stable_frames
    cfg.cooldown_seconds = args.cooldown

    if args.list_ports:
        list_serial_ports()
        return 0
    if args.send is not None:
        return run_one_shot(args.send, cfg, args.simulate)
    return run_live(cfg, args.simulate)


if __name__ == "__main__":
    sys.exit(main())
