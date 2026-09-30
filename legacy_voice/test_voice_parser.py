"""Parser tests. No microphone and no Arduino required.

Run from the repository root:

    python3 pc/test_parser.py
"""

from __future__ import annotations

import unittest

from voice_command_parser import COMMANDS, parse_command


class ParserTests(unittest.TestCase):
    def test_every_listed_phrase_maps_to_its_command(self) -> None:
        for command, phrases in COMMANDS.items():
            for phrase in phrases:
                self.assertEqual(parse_command(phrase), command, phrase)

    def test_required_version_1_phrases(self) -> None:
        self.assertEqual(parse_command("turn the light on"), "LIGHT_ON")
        self.assertEqual(parse_command("turn the light off"), "LIGHT_OFF")
        self.assertEqual(parse_command("light on"), "LIGHT_ON")
        self.assertEqual(parse_command("light off"), "LIGHT_OFF")

    def test_punctuation_case_and_extra_words(self) -> None:
        self.assertEqual(parse_command("Turn the light on!"), "LIGHT_ON")
        self.assertEqual(parse_command("please turn the light off"), "LIGHT_OFF")
        self.assertEqual(parse_command("can you turn the light on"), "LIGHT_ON")
        self.assertEqual(parse_command("LIGHT_ON"), "LIGHT_ON")
        self.assertEqual(parse_command("light_off"), "LIGHT_OFF")

    def test_french_phrases_with_accents(self) -> None:
        self.assertEqual(parse_command("allume la lumière"), "LIGHT_ON")
        self.assertEqual(parse_command("Éteins la lumière"), "LIGHT_OFF")

    def test_questions_and_lookalikes_are_not_commands(self) -> None:
        for text in (
            "",
            "   ",
            "hello",
            "what time is it",
            "is the light on",
            "the light on the table",
            "turn the radio on",
            "what's going on",
        ):
            self.assertIsNone(parse_command(text), text)

    def test_status_question_is_allowed(self) -> None:
        self.assertEqual(parse_command("what is the light status"), "STATUS")
        self.assertEqual(parse_command("light status"), "STATUS")


if __name__ == "__main__":
    unittest.main()
