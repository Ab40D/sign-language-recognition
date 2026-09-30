"""Tests for the temporal stability state machine (control/debounce.py)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.debounce import SignStabilizer  # noqa: E402

FRAME = 1.0 / 30.0  # simulate a 30 fps camera


def run_frames(stabilizer: SignStabilizer, signs: list[str | None], start: float = 0.0):
    """Feed a sequence of per-frame predictions; return all fired signs."""
    fired = []
    now = start
    for sign in signs:
        result = stabilizer.update(sign, now)
        if result is not None:
            fired.append(result)
        now += FRAME
    return fired, now


class DebounceTests(unittest.TestCase):
    def make(self) -> SignStabilizer:
        # stable_frames=5, hold>=0.10s (3 frames at 30fps), cooldown 0.5s
        return SignStabilizer(stable_frames=5, min_hold_seconds=0.10,
                              cooldown_seconds=0.5, none_reset_frames=3)

    def test_stable_hold_fires_exactly_once(self) -> None:
        stabilizer = self.make()
        fired, _ = run_frames(stabilizer, ["ON"] * 90)  # 3 seconds of ON
        self.assertEqual(fired, ["ON"])

    def test_flicker_does_not_fire(self) -> None:
        stabilizer = self.make()
        # ON stable, then 2 stray OFF frames, then ON again.
        signs = ["ON"] * 20 + ["OFF", "OFF"] + ["ON"] * 20
        fired, _ = run_frames(stabilizer, signs)
        self.assertEqual(fired, ["ON"])

    def test_sign_change_fires_again(self) -> None:
        stabilizer = self.make()
        signs = ["ON"] * 30 + ["OFF"] * 60
        fired, _ = run_frames(stabilizer, signs)
        self.assertEqual(fired, ["ON", "OFF"])

    def test_hand_leaving_resets_memory(self) -> None:
        stabilizer = self.make()
        signs = ["ON"] * 30 + [None] * 10 + ["ON"] * 30
        fired, _ = run_frames(stabilizer, signs)
        self.assertEqual(fired, ["ON", "ON"])

    def test_brief_gap_does_not_reset_memory(self) -> None:
        stabilizer = self.make()
        # 2 absent frames (< none_reset_frames=3) must not re-arm the sign.
        signs = ["ON"] * 30 + [None, None] + ["ON"] * 30
        fired, _ = run_frames(stabilizer, signs)
        self.assertEqual(fired, ["ON"])

    def test_cooldown_suppresses_too_fast_switch(self) -> None:
        # cooldown 0.8s. ON fires at ~0.13s. OFF becomes stable at ~0.83s
        # after the start, i.e. only 0.7s after the ON command -> still
        # inside the cooldown, so OFF is suppressed. Switching back to ON
        # later fires ON again.
        stabilizer = SignStabilizer(stable_frames=5, min_hold_seconds=0.10,
                                    cooldown_seconds=0.8, none_reset_frames=3)
        signs = ["ON"] * 20 + ["OFF"] * 6 + ["ON"] * 40
        fired, _ = run_frames(stabilizer, signs)
        # Only the first ON fires: the quick OFF is swallowed by the cooldown,
        # and the return to ON must not re-send ON (the physical state never
        # changed, so resending would only risk a double toggle).
        self.assertEqual(fired, ["ON"])

    def test_late_switch_still_fires_after_cooldown(self) -> None:
        # Same as above, but OFF is HELD past the cooldown -> it fires late
        # rather than being lost forever.
        stabilizer = SignStabilizer(stable_frames=5, min_hold_seconds=0.10,
                                    cooldown_seconds=0.8, none_reset_frames=3)
        signs = ["ON"] * 20 + ["OFF"] * 60
        fired, _ = run_frames(stabilizer, signs)
        self.assertEqual(fired, ["ON", "OFF"])

    def test_reset_clears_state(self) -> None:
        stabilizer = self.make()
        run_frames(stabilizer, ["ON"] * 30)
        stabilizer.reset()
        fired, _ = run_frames(stabilizer, ["ON"] * 30)
        self.assertEqual(fired, ["ON"])  # fires again after explicit reset


if __name__ == "__main__":
    unittest.main()
