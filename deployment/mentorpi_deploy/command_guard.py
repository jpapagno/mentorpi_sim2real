"""Pure, monotonic-time command gate shared by hardware and regression tests."""

import math

BODY_LIMITS = (0.1, 0.1, 0.3)


class CommandGuard:
    def __init__(self):
        self.command = (0.0, 0.0, 0.0)
        self.command_at = float("-inf")
        self.enable_at = float("-inf")
        self.scan_at = float("-inf")
        self.imu_at = float("-inf")
        self.enabled = False

    def receive_command(self, command, now):
        if len(command) != 3 or not all(math.isfinite(v) for v in command):
            self.command = (0.0, 0.0, 0.0)
            self.command_at = float("-inf")
            return
        self.command = tuple(max(-limit, min(limit, value)) for value, limit in zip(command, BODY_LIMITS, strict=True))
        self.command_at = now

    def receive_enable(self, enabled, now):
        if enabled and not self.enabled:
            # A command received while disarmed must never replay on arming.
            self.command = (0.0, 0.0, 0.0)
            self.command_at = float("-inf")
        self.enabled = enabled
        self.enable_at = now

    def output(self, now):
        reasons = []
        if not self.enabled or now - self.enable_at > 0.25:
            reasons.append("disarmed")
        if now - self.command_at > 0.20:
            reasons.append("command_stale")
        if now - self.scan_at > 0.40:
            reasons.append("scan_stale")
        if now - self.imu_at > 0.25:
            reasons.append("imu_stale")
        return ((0.0, 0.0, 0.0) if reasons else self.command), reasons
