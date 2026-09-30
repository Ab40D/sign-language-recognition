"""Map spoken or typed phrases to the Version 1 serial protocol.

The Arduino never sees these phrases. It only sees LIGHT_ON, LIGHT_OFF,
and STATUS. Keep those token names identical to docs/protocol.md and to
the sketch in arduino/voice_light_controller/.
"""

from __future__ import annotations

import re

# Tokens the PC is allowed to send. The sketch accepts exactly these.
PROTOCOL_COMMANDS = ("LIGHT_ON", "LIGHT_OFF", "STATUS")

# Longer phrases may match inside a sentence ("please turn the light on").
# One- and two-word phrases must be the whole utterance or sit at an edge,
# so "the light on the table" is not treated as a command.
COMMANDS: dict[str, tuple[str, ...]] = {
    "STATUS": (
        "status",
        "light status",
        "led status",
        "what is the light status",
    ),
    "LIGHT_OFF": (
        "turn the light off",
        "turn off the light",
        "turn the lights off",
        "turn off the lights",
        "turn the led off",
        "turn off the led",
        "switch the light off",
        "switch off the light",
        "switch the lights off",
        "switch the led off",
        "light off",
        "lights off",
        "led off",
        "lamp off",
        "turn it off",
        "switch it off",
        "turn light off",
        "shut the light off",
        "shut off the light",
        # French phrases, so --language fr-FR can work later without
        # changing the Arduino. Stored without accents; input is folded.
        "eteins la lumiere",
        "eteindre la lumiere",
        "eteint la lumiere",
        "coupe la lumiere",
        "eteins la lampe",
    ),
    "LIGHT_ON": (
        "turn the light on",
        "turn on the light",
        "turn the lights on",
        "turn on the lights",
        "turn the led on",
        "turn on the led",
        "switch the light on",
        "switch on the light",
        "switch the lights on",
        "switch the led on",
        "light on",
        "lights on",
        "led on",
        "lamp on",
        "turn it on",
        "switch it on",
        "turn light on",
        "allume la lumiere",
        "allumer la lumiere",
        "allume la lampe",
    ),
}

# Questions are not commands, except an explicit status phrase.
_QUESTION_PREFIXES = (
    "is ",
    "was ",
    "are ",
    "what ",
    "whats ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "do ",
    "does ",
    "did ",
    "can you tell",
    "could you tell",
)

_ACCENTS = str.maketrans(
    {
        "à": "a",
        "á": "a",
        "â": "a",
        "ä": "a",
        "ç": "c",
        "è": "e",
        "é": "e",
        "ê": "e",
        "ë": "e",
        "ì": "i",
        "í": "i",
        "î": "i",
        "ï": "i",
        "ò": "o",
        "ó": "o",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "u",
        "ÿ": "y",
        "œ": "oe",
        "æ": "ae",
    }
)


def normalize(text: str) -> str:
    """Lowercase, fold accents, and turn punctuation into spaces."""
    folded = text.strip().lower().translate(_ACCENTS)
    folded = folded.replace("_", " ")
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def phrase_matches(normalized: str, phrase: str) -> bool:
    if not phrase:
        return False
    if len(phrase.split()) >= 3:
        return phrase in normalized
    if normalized == phrase:
        return True
    if normalized.startswith(phrase + " "):
        return True
    return normalized.endswith(" " + phrase)


def parse_command(text: str) -> str | None:
    """Return a protocol token, or None if this should not be sent.

    None is intentional. Unrecognized speech stops on the PC. It is not
    forwarded to the Arduino for the board to reject.
    """
    normalized = normalize(text)
    if not normalized:
        return None

    if any(normalized.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        for phrase in COMMANDS["STATUS"]:
            if phrase_matches(normalized, phrase):
                return "STATUS"
        return None

    # OFF before ON is defensive. With the current phrase list they do
    # not overlap, but a later phrase might.
    for command in ("STATUS", "LIGHT_OFF", "LIGHT_ON"):
        for phrase in COMMANDS[command]:
            if phrase_matches(normalized, phrase):
                return command
    return None
