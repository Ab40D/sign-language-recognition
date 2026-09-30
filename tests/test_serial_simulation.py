"""Tests for the simulated Arduino (control/serial_controller.py)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.serial_controller import SimulatedSerialController  # noqa: E402


class SimulatedSerialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = SimulatedSerialController()
        self.sim.connect()

    def test_light_on_and_off(self) -> None:
        self.assertEqual(self.sim.send_command("LIGHT_ON"), "LIGHT_STATE:ON")
        self.assertEqual(self.sim.send_command("LIGHT_OFF"), "LIGHT_STATE:OFF")

    def test_status_tracks_state(self) -> None:
        self.assertEqual(self.sim.send_command("STATUS"), "LIGHT_STATE:OFF")
        self.sim.send_command("LIGHT_ON")
        self.assertEqual(self.sim.send_command("STATUS"), "LIGHT_STATE:ON")

    def test_unknown_command_rejected_like_the_sketch(self) -> None:
        self.assertEqual(self.sim.send_command("FLY_TO_MOON"), "ERROR:UNKNOWN_COMMAND")
        self.assertNotIn("FLY_TO_MOON", self.sim.commands_received)

    def test_invalid_token_is_refused(self) -> None:
        self.assertEqual(self.sim.send_command("bad;token"), "")
        self.assertEqual(self.sim.commands_received, [])

    def test_status_text_is_simulation(self) -> None:
        self.assertEqual(self.sim.status_text(), "SIMULATION")


if __name__ == "__main__":
    unittest.main()
