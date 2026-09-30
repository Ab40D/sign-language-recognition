"""Tests for the sign -> command mapping layer."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.command_parser import is_valid_command, sign_to_command  # noqa: E402


class CommandParserTests(unittest.TestCase):
    def test_signs_map_to_protocol_tokens(self) -> None:
        self.assertEqual(sign_to_command("ON"), "LIGHT_ON")
        self.assertEqual(sign_to_command("OFF"), "LIGHT_OFF")

    def test_no_sign_means_no_command(self) -> None:
        self.assertIsNone(sign_to_command(None))
        self.assertIsNone(sign_to_command("UNKNOWN"))
        self.assertIsNone(sign_to_command("garbage"))

    def test_protocol_validation(self) -> None:
        self.assertTrue(is_valid_command("LIGHT_ON"))
        self.assertTrue(is_valid_command("LIGHT_OFF"))
        self.assertTrue(is_valid_command("STATUS"))
        self.assertFalse(is_valid_command("LIGHT_ON; DROP TABLE"))
        self.assertFalse(is_valid_command("light_on"))
        self.assertFalse(is_valid_command(""))


if __name__ == "__main__":
    unittest.main()
