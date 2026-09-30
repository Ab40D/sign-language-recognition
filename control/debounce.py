"""Temporal stability: only fire a command when a sign is truly held.

Problem: the classifier runs every frame (~30 fps). If the user holds the
ON sign for five seconds we must send LIGHT_ON exactly once — not 150
times. And flickers (one stray "OFF" frame between two "ON" frames) must
never reach the Arduino.

Solution, implemented as a small state machine (SignStabilizer):

  * consecutive-frame confirmation — the same sign must be predicted in
    `stable_frames` consecutive frames, AND
  * time-based stability           — it must have been held for at least
    `min_hold_seconds`, AND
  * state-change detection         — a command only fires when the stable
    sign differs from the last one we sent, AND
  * cooldown                       — after firing we ignore triggers for
    `cooldown_seconds`.

While the hand disappears for `none_reset_frames` frames, the "last sent"
memory clears, so showing the same sign again later legitimately sends it
again.
"""

from __future__ import annotations


class SignStabilizer:
    def __init__(
        self,
        stable_frames: int = 6,
        min_hold_seconds: float = 0.5,
        cooldown_seconds: float = 1.0,
        none_reset_frames: int = 4,
    ) -> None:
        self.stable_frames = stable_frames
        self.min_hold_seconds = min_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.none_reset_frames = none_reset_frames
        self.reset()

    def reset(self) -> None:
        """Forget everything (used by the C key and at startup)."""
        self.candidate: str | None = None
        self.candidate_count = 0
        self.candidate_first_seen = 0.0
        self.none_streak = 0
        self.last_triggered: str | None = None
        self.last_send_time = float("-inf")

    def update(self, sign: str | None, now: float) -> str | None:
        """Feed one frame's recognition result.

        sign: "ON", "OFF", or None (no hand / UNKNOWN / low confidence).
        now:  current time in seconds (time.monotonic()).

        Returns the sign to fire ("ON"/"OFF") exactly once per stable
        transition, otherwise None.
        """
        if sign is None:
            # No usable sign this frame. Break any candidate streak.
            self.candidate = None
            self.candidate_count = 0
            self.none_streak += 1
            if self.none_streak >= self.none_reset_frames:
                # Hand has really left: allow the same sign to fire again
                # the next time it appears.
                self.last_triggered = None
            return None

        self.none_streak = 0
        if sign == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = sign
            self.candidate_count = 1
            self.candidate_first_seen = now

        held_long_enough = (now - self.candidate_first_seen) >= self.min_hold_seconds
        stable = self.candidate_count >= self.stable_frames and held_long_enough
        if not stable:
            return None
        if sign == self.last_triggered:
            return None  # already sent; holding the same sign sends nothing
        if now - self.last_send_time < self.cooldown_seconds:
            return None  # still cooling down from the previous command

        self.last_triggered = sign
        self.last_send_time = now
        return sign
