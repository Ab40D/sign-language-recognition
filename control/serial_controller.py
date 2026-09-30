"""USB serial link between Python and the Arduino.

Most of this module is reused, almost unchanged, from the original voice
project (legacy_voice/voice_controller.py): the ArduinoLink class, the
port auto-detection scoring, and the human-readable open-error messages
proved themselves there, so they were kept instead of rewritten.

What was added for the ASL project:
  * SimulatedSerialController  -> --simulate mode, no hardware needed
  * automatic reconnection     -> a pulled USB cable no longer kills the app
"""

from __future__ import annotations

import re
import time
from typing import Any

# Protocol lines the sketch may print. Keep in sync with the .ino file.
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


def require_serial() -> Any:
    """Import pyserial lazily so --simulate works without it installed."""
    try:
        import serial
        import serial.tools.list_ports  # noqa: F401  (not auto-imported)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pyserial is not installed. Run:  python -m pip install -r requirements.txt"
        ) from exc
    return serial


def explain_open_error(port: str, exc: BaseException) -> str:
    """Turn a raw SerialException into an actionable message (reused)."""
    text = str(exc).lower()
    cause = exc.__cause__ or exc
    errno = getattr(cause, "errno", None)
    if errno == 13 or "permission denied" in text:
        return (
            f"Permission denied opening {port}.\n"
            "On Linux the serial group is 'dialout'. Add your user, then log\n"
            "out and back in:\n"
            "    sudo usermod -aG dialout $USER\n"
            f"Quick fix for the current session:  sudo chmod a+rw {port}"
        )
    if errno == 16 or "busy" in text or "resource busy" in text:
        return (
            f"{port} is already open. Close the Arduino IDE Serial Monitor\n"
            f"(or any other program using it), then retry.  fuser -v {port}\n"
            "shows who holds the port on Linux."
        )
    if errno == 2 or "no such file" in text or "could not open port" in text:
        return (
            f"{port} does not exist. Unplug the board, plug it back in, and\n"
            "run:  python main.py --list-ports\n"
            "If the power LED is on but no port appears, the USB cable is\n"
            "probably charge-only. Try a data cable."
        )
    return f"Could not open {port}: {exc}"


def score_port(port: Any) -> int:
    """Rank how likely a serial port is an Arduino (reused unchanged)."""
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


def list_serial_ports() -> None:
    """Print all serial ports with enough detail to pick one."""
    serial_module = require_serial()
    ports = list(serial_module.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        print("Plug in the Arduino with a DATA USB cable, then run this again.")
        return
    print("Serial ports:")
    for port in ports:
        vid = f"{port.vid:04X}" if port.vid is not None else "----"
        pid = f"{port.pid:04X}" if port.pid is not None else "----"
        name = port.description or port.manufacturer or "unknown device"
        print(f"  {port.device:<18} {name}   VID:PID {vid}:{pid}")


def choose_port(requested: str | None) -> str:
    """Use the requested port, or auto-detect the best Arduino candidate."""
    if requested:
        return requested
    serial_module = require_serial()
    ranked = sorted(
        serial_module.tools.list_ports.comports(), key=score_port, reverse=True
    )
    if not ranked or score_port(ranked[0]) <= 0:
        raise RuntimeError(
            "No Arduino serial port found.\n"
            "Plug in the board, then run:  python main.py --list-ports\n"
            "Or pass one explicitly:      python main.py --port /dev/ttyACM0"
        )
    best_score = score_port(ranked[0])
    tied = [port for port in ranked if score_port(port) == best_score]
    if len(tied) > 1:
        lines = "\n".join(f"  {port.device}  {port.description}" for port in tied)
        raise RuntimeError(
            "More than one likely Arduino port. Pass one with --port:\n" + lines
        )
    return ranked[0].device


class ArduinoLink:
    """One USB serial session with the board (reused from the voice project).

    Opening the port asserts DTR. On an Uno R3 that resets the board, so
    after open() the sketch restarts and prints READY. Do not flush the
    input buffer before READY arrives — that line is the signal that
    setup() has finished.
    """

    def __init__(self, port: str, baud: int, ready_timeout: float, response_timeout: float) -> None:
        self.port = port
        self.baud = baud
        self.ready_timeout = ready_timeout
        self.response_timeout = response_timeout
        self.ser: Any = None

    def open(self) -> None:
        serial_module = require_serial()
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
            raise RuntimeError(explain_open_error(self.port, exc)) from exc

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def wait_until_ready(self) -> str:
        """Read lines until READY (or timeout). Returns the line seen."""
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            line = self._read_line(deadline)
            if line in PROTOCOL_LINES:
                return line
        return ""

    def send(self, command: str) -> str:
        """Send one protocol token and wait for the board's one-line reply.

        Returns the reply string ('' on timeout). Raises SerialException
        (or OSError) if the cable was pulled; the caller decides what to do.
        """
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("serial port is not open")
        payload = (command + "\n").encode("ascii")
        # Drop late boot noise so it cannot be mistaken for the reply.
        self.ser.reset_input_buffer()
        self.ser.write(payload)
        self.ser.flush()
        deadline = time.monotonic() + self.response_timeout
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


class SerialController:
    """High-level, crash-resistant wrapper around ArduinoLink.

    The real-time loop calls send_command(). If the board disappears the
    controller marks itself disconnected, keeps the app alive, and retries
    in the background (every reconnect_interval seconds).
    """

    def __init__(self, port: str, baud: int, ready_timeout: float,
                 response_timeout: float, reconnect_interval: float) -> None:
        self.port = port
        self.baud = baud
        self.ready_timeout = ready_timeout
        self.response_timeout = response_timeout
        self.reconnect_interval = reconnect_interval
        self.link: ArduinoLink | None = None
        self.connected = False
        self.last_response = ""
        self._next_reconnect_at = 0.0

    def connect(self) -> bool:
        """Open the port and wait for READY. Never raises for I/O problems."""
        try:
            link = ArduinoLink(self.port, self.baud, self.ready_timeout, self.response_timeout)
            link.open()
            ready = link.wait_until_ready()
        except (RuntimeError, OSError) as exc:
            print(f"[serial] connect failed: {exc}")
            self._schedule_reconnect()
            return False
        self.link = link
        self.connected = True
        if ready != "READY":
            print(f"[serial] no READY line (got {ready!r}); continuing anyway")
        return True

    def send_command(self, command: str) -> str:
        """Send one command. Returns the reply ('' if it failed)."""
        if not TOKEN_RE.fullmatch(command):
            # Refuse to send anything that is not a protocol token.
            print(f"[serial] refusing to send invalid token {command!r}")
            return ""
        if self.link is None:
            return ""
        try:
            response = self.link.send(command)
        except (OSError, RuntimeError) as exc:
            print(f"[serial] connection lost ({exc}); will reconnect")
            self._drop_link()
            return ""
        self.last_response = response
        if not response:
            # A silent board is suspicious; verify before trusting it.
            print("[serial] no response (timeout)")
        return response

    def maybe_reconnect(self) -> None:
        """Called every loop iteration; retries a dropped port in the background."""
        if self.connected or time.monotonic() < self._next_reconnect_at:
            return
        print(f"[serial] retrying connection on {self.port}...")
        self.connect()

    def status_text(self) -> str:
        return "CONNECTED" if self.connected else "DISCONNECTED"

    def close(self) -> None:
        if self.link is not None:
            self.link.close()
            self.link = None
        self.connected = False

    def _drop_link(self) -> None:
        if self.link is not None:
            self.link.close()
        self.link = None
        self.connected = False
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        self._next_reconnect_at = time.monotonic() + self.reconnect_interval


class SimulatedSerialController:
    """Same interface as SerialController, no hardware.

    It answers exactly like the sketch would, so the whole pipeline can be
    exercised on a machine without an Arduino. The UI shows SIMULATION.
    """

    def __init__(self) -> None:
        self.connected = True
        self.last_response = ""
        self.state_on = False
        self.commands_received: list[str] = []  # useful for tests

    def connect(self) -> bool:
        return True

    def send_command(self, command: str) -> str:
        if not TOKEN_RE.fullmatch(command):
            print(f"[simulate] refusing to send invalid token {command!r}")
            return ""
        if command in ("LIGHT_ON", "LIGHT_OFF"):
            self.commands_received.append(command)
        if command == "LIGHT_ON":
            self.state_on = True
            response = "LIGHT_STATE:ON"
        elif command == "LIGHT_OFF":
            self.state_on = False
            response = "LIGHT_STATE:OFF"
        elif command == "STATUS":
            response = "LIGHT_STATE:ON" if self.state_on else "LIGHT_STATE:OFF"
        else:
            response = "ERROR:UNKNOWN_COMMAND"
        self.last_response = response
        return response

    def maybe_reconnect(self) -> None:
        return None

    def status_text(self) -> str:
        return "SIMULATION"

    def close(self) -> None:
        self.connected = False
