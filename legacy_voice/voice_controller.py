#!/usr/bin/env python3
"""PC side of the Version 1 voice-controlled LED.

Listens on the microphone, maps a short phrase to a protocol token, and
writes that token to an Arduino Uno R3 over USB serial. The spoken
sentence is never sent to the board.

Examples, from the repository root:

    python3 pc/voice_controller.py --list-ports
    python3 pc/voice_controller.py --send LIGHT_ON
    python3 pc/voice_controller.py --parse "turn the light on"
    python3 pc/voice_controller.py
    python3 pc/voice_controller.py --mode text
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from typing import Any

from voice_command_parser import PROTOCOL_COMMANDS, parse_command

BAUD_DEFAULT = 9600
READY_TIMEOUT_S = 4.0
RESPONSE_TIMEOUT_S = 2.0
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHUNK_MS = 30
CHUNK_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * SAMPLE_WIDTH

PROTOCOL_LINES = {
    "READY",
    "LIGHT_STATE:ON",
    "LIGHT_STATE:OFF",
    "ERROR:UNKNOWN_COMMAND",
    "ERROR:LINE_TOO_LONG",
}

# Official Uno R3 uses 2341:0043. Clones often use a CH340, CP210x, or FTDI.
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403}
TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,30}$")


def say(text: str = "") -> None:
    print(text, flush=True)


class ArduinoLink:
    """One USB serial session with the Uno.

    Opening the port asserts DTR. On an Uno R3 that resets the board.
    Do not flush the input buffer before READY arrives — that line is
    the signal that setup() has finished.
    """

    def __init__(self, port: str, baud: int, verbose: bool = False) -> None:
        self.port = port
        self.baud = baud
        self.verbose = verbose
        self.serial_module: Any = None
        self.ser: Any = None

    def open(self) -> None:
        serial_module, _ = require_serial()
        self.serial_module = serial_module
        try:
            self.ser = serial_module.Serial(
                self.port,
                self.baud,
                timeout=1,
                bytesize=serial_module.EIGHTBITS,
                parity=serial_module.PARITY_NONE,
                stopbits=serial_module.STOPBITS_ONE,
            )
        except Exception as exc:
            raise SystemExit(explain_open_error(self.port, exc)) from exc

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def wait_until_ready(self, timeout: float = READY_TIMEOUT_S) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._read_line(deadline)
            if line in PROTOCOL_LINES:
                return line
            if self.verbose and line:
                say(f"(ignored while waiting) {line!r}")
        return ""

    def send(self, command: str) -> str:
        if self.ser is None:
            raise RuntimeError("serial port is not open")
        payload = (command + "\n").encode("ascii")
        # Drop late boot noise so it cannot be mistaken for the reply.
        self.ser.reset_input_buffer()
        self.ser.write(payload)
        self.ser.flush()
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        while time.monotonic() < deadline:
            line = self._read_line(deadline)
            if line:
                return line
        return ""

    def _read_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ""
        self.ser.timeout = max(0.05, remaining)
        raw = self.ser.readline()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace").strip()


def require_serial() -> tuple[Any, Any]:
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise SystemExit(
            "pyserial is not installed.\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  python -m pip install -r pc/requirements.txt"
        ) from exc
    return serial, list_ports


def explain_open_error(port: str, exc: BaseException) -> str:
    text = str(exc).lower()
    cause = exc.__cause__ or exc
    errno = getattr(cause, "errno", None)
    if errno == 13 or "permission denied" in text:
        return (
            f"Permission denied opening {port}.\n\n"
            "The serial device group is dialout. Add your user, then log out\n"
            "and back in (a new terminal is not always enough):\n\n"
            '  sudo usermod -aG dialout "$USER"\n'
            "  id -nG | tr ' ' '\\n' | grep dialout\n\n"
            "Until you log in again, this works for the current plug-in only:\n\n"
            f"  sudo chmod a+rw {port}\n"
        )
    if errno == 16 or "busy" in text or "resource busy" in text:
        return (
            f"{port} is already open.\n\n"
            "Close the Arduino IDE Serial Monitor, then check who holds it:\n\n"
            f"  fuser -v {port}\n"
        )
    if errno == 2 or "no such file" in text or "could not open port" in text:
        return (
            f"{port} does not exist.\n\n"
            "Unplug the board, plug it back in, and run:\n\n"
            "  python3 pc/voice_controller.py --list-ports\n\n"
            "If the Uno's power LED is on but no port appears, the cable is\n"
            "probably charge-only. Use a data cable.\n"
        )
    return f"Could not open {port}: {exc}"


def score_port(port: Any) -> int:
    description = " ".join(
        part
        for part in (port.description, port.manufacturer, port.product, port.device)
        if part
    ).lower()
    device = port.device or ""
    score = 0
    if getattr(port, "vid", None) in (0x2341, 0x2A03):
        score += 100
    if "arduino" in description:
        score += 80
    if getattr(port, "vid", None) in ARDUINO_VIDS:
        score += 40
    if device.startswith("/dev/ttyACM"):
        score += 25
    elif device.startswith("/dev/ttyUSB"):
        score += 15
    if "bluetooth" in description or "/dev/ttyS" in device:
        score -= 80
    return score


def list_serial_ports() -> int:
    _, list_ports = require_serial()
    ports = list(list_ports.comports())
    if not ports:
        say("No serial ports found.")
        say("Plug in the Uno with a data cable, then run this command again.")
        return 1
    say("Serial ports:")
    for port in ports:
        vid = f"{port.vid:04X}" if port.vid is not None else "----"
        pid = f"{port.pid:04X}" if port.pid is not None else "----"
        name = port.description or port.manufacturer or "unknown device"
        say(f"  {port.device:<18} {name}   VID:PID {vid}:{pid}")
    return 0


def choose_port(requested: str | None) -> str:
    if requested:
        return requested
    env_port = os.environ.get("VOICE_ARDUINO_PORT")
    if env_port:
        return env_port
    _, list_ports = require_serial()
    ranked = sorted(list_ports.comports(), key=score_port, reverse=True)
    if not ranked or score_port(ranked[0]) <= 0:
        raise SystemExit(
            "No Arduino serial port found.\n"
            "Plug in the Uno and run:  python3 pc/voice_controller.py --list-ports"
        )
    best = score_port(ranked[0])
    tied = [port for port in ranked if score_port(port) == best]
    if len(tied) > 1:
        lines = "\n".join(f"  {port.device}  {port.description}" for port in tied)
        raise SystemExit(
            "More than one likely Arduino port. Pass one with --port:\n" + lines
        )
    return ranked[0].device


def normalize_outgoing(command: str) -> str:
    token = command.strip().upper()
    if not TOKEN_RE.fullmatch(token):
        known = ", ".join(PROTOCOL_COMMANDS)
        raise SystemExit(
            f"Refusing to send {command!r}.\n"
            f"Use an uppercase token such as {known}, or NOPE to test the error path."
        )
    return token


def print_result(command: str, response: str) -> int:
    say(f"Command: {command}\n")
    say("Sending to Arduino...\n")
    if not response:
        say("Arduino: (no response — timeout)")
        say("Check the baud rate, that the sketch is uploaded, and that")
        say("the Serial Monitor is closed.")
        return 1
    say(f"Arduino: {response}")
    if response.startswith("LIGHT_STATE:"):
        return 0
    return 1


def open_link(port: str, baud: int, verbose: bool) -> ArduinoLink:
    say(f"Port: {port}    Baud: {baud}")
    say("Waiting for Arduino READY...")
    link = ArduinoLink(port, baud, verbose=verbose)
    link.open()
    ready = link.wait_until_ready()
    if ready == "READY":
        say("Arduino: READY\n")
    elif ready:
        say(f"Arduino is already running ({ready}).\n")
    else:
        say("No READY line within 4 seconds.")
        say("The sketch may not be uploaded, or the baud rate is not 9600.")
        say("Continuing so you can still try one command.\n")
    return link


def transcribe(audio: Any, engine: str, language: str, vosk_model: str | None) -> str:
    if engine == "google":
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        try:
            text = recognizer.recognize_google(audio, language=language)
        except AttributeError as exc:
            raise SystemExit(
                "This SpeechRecognition build has no recognize_google.\n"
                "Install 3.17 or newer:  python -m pip install 'SpeechRecognition>=3.17,<4'"
            ) from exc
        return str(text).strip()

    if engine == "vosk":
        return transcribe_vosk(audio, vosk_model)

    raise SystemExit(f"Unknown speech engine: {engine}")


_VOSK_MODEL: Any = None


def transcribe_vosk(audio: Any, model_path: str | None) -> str:
    global _VOSK_MODEL
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError as exc:
        raise SystemExit(
            "vosk is not installed.\n"
            "  python -m pip install -r pc/requirements-offline.txt"
        ) from exc

    if not model_path or not os.path.isdir(model_path):
        raise SystemExit(
            "Vosk needs an unpacked model directory.\n\n"
            "  mkdir -p \"$HOME/vosk-models\" && cd \"$HOME/vosk-models\"\n"
            "  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
            "  unzip vosk-model-small-en-us-0.15.zip\n\n"
            "Then pass:\n"
            "  --engine vosk --vosk-model \"$HOME/vosk-models/vosk-model-small-en-us-0.15\""
        )

    SetLogLevel(-1)
    if _VOSK_MODEL is None:
        say("Loading Vosk model...")
        _VOSK_MODEL = Model(model_path)
    pcm = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=SAMPLE_WIDTH)
    recognizer = KaldiRecognizer(_VOSK_MODEL, SAMPLE_RATE)
    recognizer.AcceptWaveform(pcm)
    text = json.loads(recognizer.FinalResult()).get("text", "")
    return str(text).strip()


def audio_from_pcm(pcm: bytes) -> Any:
    import speech_recognition as sr

    return sr.AudioData(pcm, SAMPLE_RATE, SAMPLE_WIDTH)


def pyaudio_available() -> bool:
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        return False
    return True


def list_mics() -> int:
    say("ALSA capture devices:")
    try:
        completed = subprocess.run(["arecord", "-l"], check=False)
        if completed.returncode != 0:
            say("arecord found no capture devices.")
    except FileNotFoundError:
        say("arecord is not installed.  sudo apt install alsa-utils")
    say("")
    if not pyaudio_available():
        say("PyAudio is not installed, so there is no device index list.")
        say("Voice mode can use --input arecord.")
        return 0
    import speech_recognition as sr

    say("PyAudio devices (pass the index to --mic):")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        say(f"  {index}: {name}")
    return 0


def rms_int16(data: bytes) -> float:
    count = len(data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack("<" + "h" * count, data[: count * 2])
    return math.sqrt(sum(sample * sample for sample in samples) / count)


def read_exact(stream: Any, size: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < size:
        piece = stream.read(size - len(buf))
        if not piece:
            return None
        buf.extend(piece)
    return bytes(buf)


def start_arecord(device: str | None) -> subprocess.Popen[bytes]:
    command = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(SAMPLE_RATE),
        "-t",
        "raw",
    ]
    if device:
        command.extend(["-D", device])
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "arecord was not found. Install it with:\n  sudo apt install alsa-utils"
        ) from exc


def stop_arecord(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def measure_ambient(device: str | None, verbose: bool) -> float:
    """Sample room noise once. Do not do this inside the listen loop."""
    process = start_arecord(device)
    assert process.stdout is not None
    levels: list[float] = []
    try:
        for _ in range(16):
            chunk = read_exact(process.stdout, CHUNK_BYTES)
            if chunk is None:
                raise SystemExit(arecord_failure(process))
            levels.append(rms_int16(chunk))
    finally:
        stop_arecord(process)
    noise = sorted(levels)[len(levels) // 2]
    threshold = max(400.0, noise * 3.0)
    if verbose:
        say(f"Ambient RMS {noise:.0f}, speech threshold {threshold:.0f}")
    return threshold


def record_arecord(device: str | None, threshold: float) -> bytes:
    process = start_arecord(device)
    assert process.stdout is not None
    preroll: list[bytes] = []
    speech: list[bytes] = []
    started = False
    silent_chunks = 0
    waited = 0
    max_wait = 12000 // CHUNK_MS
    max_speech = 5000 // CHUNK_MS
    end_silence = 700 // CHUNK_MS
    try:
        while True:
            chunk = read_exact(process.stdout, CHUNK_BYTES)
            if chunk is None:
                break
            level = rms_int16(chunk)
            if not started:
                preroll.append(chunk)
                preroll = preroll[-4:]
                waited += 1
                if level >= threshold:
                    started = True
                    speech.extend(preroll)
                elif waited >= max_wait:
                    return b""
            else:
                speech.append(chunk)
                silent_chunks = silent_chunks + 1 if level < threshold else 0
                if silent_chunks >= end_silence or len(speech) >= max_speech:
                    break
        return b"".join(speech)
    finally:
        stop_arecord(process)


def arecord_failure(process: subprocess.Popen[bytes]) -> str:
    stop_arecord(process)
    detail = ""
    if process.stderr is not None:
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
    hint = (
        "Could not read from the microphone with arecord.\n"
        "List devices with:  arecord -l\n"
        "Then pass one, for example:  --alsa-device plughw:0,0\n"
        "On PipeWire, also try:  sudo apt install pipewire-alsa"
    )
    if detail:
        return hint + "\n\n" + detail
    return hint


def resolve_input(requested: str) -> str:
    if requested == "pyaudio":
        if not pyaudio_available():
            raise SystemExit(
                "PyAudio is not installed.\n"
                "  sudo apt install portaudio19-dev python3-dev build-essential\n"
                "  python -m pip install -r pc/requirements-pyaudio.txt\n\n"
                "Or run without it:  --input arecord"
            )
        return "pyaudio"
    if requested == "arecord":
        return "arecord"
    if pyaudio_available():
        return "pyaudio"
    say("PyAudio is not installed. Using arecord for the microphone.")
    return "arecord"


def handle_transcript(link: ArduinoLink, text: str, from_voice: bool) -> None:
    say(f"You said: {text}\n")
    if from_voice:
        command = parse_command(text)
    else:
        token = text.strip().upper()
        command = token if token in PROTOCOL_COMMANDS else parse_command(text)
    if command is None:
        say("Command: (not recognized)")
        say('Try "turn the light on" or "turn the light off". Nothing was sent.\n')
        return
    response = link.send(command)
    print_result(command, response)
    say("")


def run_voice(args: argparse.Namespace, link: ArduinoLink) -> int:
    input_mode = resolve_input(args.input)
    engine_line = (
        f"Speech: Google Web Speech (online, {args.language})"
        if args.engine == "google"
        else "Speech: Vosk (offline)"
    )
    say("Voice Light Controller — Version 1")
    say(engine_line)
    if args.engine == "google":
        say("Audio from this engine is sent to Google. Use --engine vosk to stay offline.")
    say(f"Input: {input_mode}")
    say("")

    if input_mode == "pyaudio":
        return voice_loop_pyaudio(args, link)
    return voice_loop_arecord(args, link)


def voice_loop_pyaudio(args: argparse.Namespace, link: ArduinoLink) -> int:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8
    if args.energy is not None:
        recognizer.energy_threshold = args.energy
        recognizer.dynamic_energy_threshold = False

    try:
        # Do not force 16 kHz. Some Linux devices reject it; the recognizer
        # resamples whatever the device actually delivers.
        with sr.Microphone(device_index=args.mic) as source:
            if args.energy is None:
                say("Calibrating microphone... stay quiet for a moment.")
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                say("")
            while True:
                say("Listening...\n")
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=5,
                    )
                except sr.WaitTimeoutError:
                    continue
                try:
                    text = transcribe(audio, args.engine, args.language, args.vosk_model)
                except sr.UnknownValueError:
                    say("Could not understand the audio. Try again.\n")
                    continue
                except sr.RequestError as exc:
                    say(f"Speech service error: {exc}")
                    say("The Google engine needs internet. Or use --engine vosk.\n")
                    continue
                if not text:
                    say("Could not understand the audio. Try again.\n")
                    continue
                handle_transcript(link, text, from_voice=True)
    except KeyboardInterrupt:
        say("\nStopped.")
        return 0
    except OSError as exc:
        raise SystemExit(
            f"Microphone error: {exc}\n"
            "Run --list-mics, then try --mic INDEX or --input arecord."
        ) from exc
    return 0


def voice_loop_arecord(args: argparse.Namespace, link: ArduinoLink) -> int:
    import speech_recognition as sr

    threshold = args.threshold
    if threshold is None:
        say("Calibrating microphone... stay quiet for a moment.")
        threshold = measure_ambient(args.alsa_device, args.verbose)
        say("")
    try:
        while True:
            say("Listening...\n")
            pcm = record_arecord(args.alsa_device, threshold)
            if not pcm:
                continue
            audio = audio_from_pcm(pcm)
            try:
                text = transcribe(audio, args.engine, args.language, args.vosk_model)
            except sr.UnknownValueError:
                say("Could not understand the audio. Try again.\n")
                continue
            except sr.RequestError as exc:
                say(f"Speech service error: {exc}")
                say("The Google engine needs internet. Or use --engine vosk.\n")
                continue
            if not text:
                say("Could not understand the audio. Try again.\n")
                continue
            handle_transcript(link, text, from_voice=True)
    except KeyboardInterrupt:
        say("\nStopped.")
        return 0
    return 0


def run_text(link: ArduinoLink) -> int:
    say("Type a phrase or LIGHT_ON / LIGHT_OFF / STATUS. Type quit to stop.\n")
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                say("")
                break
            if line.strip().lower() in {"quit", "exit"}:
                break
            if not line.strip():
                continue
            say("")
            handle_transcript(link, line, from_voice=False)
    except KeyboardInterrupt:
        say("\nStopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Version 1: voice or text to an Arduino Uno LED over USB serial."
    )
    parser.add_argument("--port", help="Serial device, for example /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=None, help="Default 9600.")
    parser.add_argument("--engine", choices=("google", "vosk"), default="google")
    parser.add_argument(
        "--language",
        default=os.environ.get("VOICE_LANGUAGE", "en-US"),
        help="BCP-47 language for the Google engine. Default en-US.",
    )
    parser.add_argument(
        "--vosk-model",
        default=os.environ.get("VOSK_MODEL"),
        help="Unpacked Vosk model directory.",
    )
    parser.add_argument(
        "--input",
        choices=("auto", "pyaudio", "arecord"),
        default="auto",
        help="Microphone capture. auto uses PyAudio if it is installed.",
    )
    parser.add_argument("--mic", type=int, default=None, help="PyAudio device index.")
    parser.add_argument(
        "--alsa-device",
        help="ALSA device for arecord, for example plughw:0,0.",
    )
    parser.add_argument("--energy", type=float, help="Fixed PyAudio energy threshold.")
    parser.add_argument("--threshold", type=float, help="Fixed arecord RMS threshold.")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--parse", metavar="TEXT", help="Parse one sentence and exit.")
    parser.add_argument("--send", metavar="COMMAND", help="Send one token and exit.")
    parser.add_argument("--mode", choices=("voice", "text"), default="voice")
    parser.add_argument("--verbose", action="store_true")
    return parser


def baud_from(args: argparse.Namespace) -> int:
    if args.baud is not None:
        return args.baud
    raw = os.environ.get("VOICE_ARDUINO_BAUD", str(BAUD_DEFAULT))
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Bad VOICE_ARDUINO_BAUD value: {raw}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        return list_serial_ports()
    if args.list_mics:
        return list_mics()
    if args.parse is not None:
        say(f"You said: {args.parse}\n")
        command = parse_command(args.parse)
        if command is None:
            say("Command: (not recognized)")
            return 2
        say(f"Command: {command}")
        return 0

    # Voice mode imports SpeechRecognition only when a phrase must be heard
    # or recognized. --send and --mode text do not need it.
    if args.mode == "voice" and args.send is None:
        try:
            import speech_recognition  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "SpeechRecognition is not installed.\n"
                "  python3 -m venv .venv && source .venv/bin/activate\n"
                "  python -m pip install -r pc/requirements.txt"
            ) from exc

    port = choose_port(args.port)
    baud = baud_from(args)
    link = open_link(port, baud, args.verbose)
    try:
        if args.send is not None:
            command = normalize_outgoing(args.send)
            return print_result(command, link.send(command))
        if args.mode == "text":
            return run_text(link)
        return run_voice(args, link)
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
