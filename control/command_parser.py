"""Map recognized ASL signs to serial protocol commands.

This is the ONLY place where a sign becomes a device command. The
recognition layer knows nothing about pins or serial ports, and the
serial layer knows nothing about signs. That separation keeps each part
replaceable.

The Arduino never sees "ON"/"OFF". It only sees protocol tokens:
LIGHT_ON, LIGHT_OFF, STATUS — identical to the original voice project,
so the wire protocol stays stable while the input method changes.

This module replaces legacy_voice/voice_command_parser.py, which mapped
*spoken phrases* to the same tokens. The mapping idea survived; the
voice phrases did not.
"""

from __future__ import annotations

from config import LABEL_OFF, LABEL_ON, PROTOCOL_COMMANDS

# Sign label -> protocol token.
SIGN_TO_COMMAND: dict[str, str] = {
    LABEL_ON: "LIGHT_ON",
    LABEL_OFF: "LIGHT_OFF",
}


def sign_to_command(sign: str | None) -> str | None:
    """Return the protocol command for a sign, or None.

    None means "send nothing" (no hand, unknown sign, low confidence).
    """
    if sign is None:
        return None
    return SIGN_TO_COMMAND.get(sign)


def is_valid_command(token: str) -> bool:
    """True if a token is part of the serial protocol."""
    return token in PROTOCOL_COMMANDS
