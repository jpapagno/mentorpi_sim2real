"""Shared, dependency-free mecanum geometry in verified motor-channel order.

Body axes: +x forward, +y left, +yaw counterclockwise. Roller geometry assumes
the standard X arrangement; ground strafe/yaw commissioning is still required.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DriveConfig:
    wheel_diameter: float = 0.065
    wheelbase: float = 0.1368
    track_width: float = 0.1446
    max_wheel_rps: float = 0.8
    wheel_acceleration_rps2: float = 2.0
    body_acceleration_limits: tuple = (0.5, 0.5, 1.5)
    command_gain: float = 1.0

    def __post_init__(self):
        values = (
            self.wheel_diameter,
            self.wheelbase,
            self.track_width,
            self.max_wheel_rps,
            self.wheel_acceleration_rps2,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Drive dimensions and limits must be finite and positive")
        if len(self.body_acceleration_limits) != 3 or not all(
            math.isfinite(v) and v > 0 for v in self.body_acceleration_limits
        ):
            raise ValueError("Three positive body acceleration limits required")
        if not math.isfinite(self.command_gain) or not 1 <= self.command_gain <= 2:
            raise ValueError("Motor command gain must be in [1,2]")

    @property
    def inverse_matrix(self):
        """Body m/s, m/s, rad/s -> signed motor RPS, channels 1..4."""
        k = (self.wheelbase + self.track_width) / 2
        c = math.pi * self.wheel_diameter
        return tuple(tuple(v / c for v in row) for row in ((-1, 1, k), (-1, -1, k), (1, -1, k), (1, 1, k)))

    @property
    def forward_matrix(self):
        """Signed motor RPS -> least-squares ideal body velocity."""
        c = math.pi * self.wheel_diameter / 4
        k = (self.wheelbase + self.track_width) / 2
        return ((-c, -c, c, c), (c, -c, -c, c), (c / k, c / k, c / k, c / k))

    def wheel_speeds(self, body):
        if len(body) != 3 or not all(math.isfinite(v) for v in body):
            raise ValueError("Three finite body velocities required")
        values = tuple(
            self.command_gain * sum(a * b for a, b in zip(row, body, strict=True)) for row in self.inverse_matrix
        )
        # Scale the entire vector, preserving the requested motion direction.
        ratio = max(1.0, max(abs(v) for v in values) / self.max_wheel_rps)
        return tuple(v / ratio for v in values)

    def body_velocity(self, wheels):
        if len(wheels) != 4 or not all(math.isfinite(v) for v in wheels):
            raise ValueError("Four finite wheel speeds required")
        return tuple(sum(a * b for a, b in zip(row, wheels, strict=True)) for row in self.forward_matrix)

    def advance(self, previous, target, dt, stop=False):
        """Apply motor slew limits with an immediate zero on stop."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite dt required")
        if len(previous) != 4 or len(target) != 4 or not all(math.isfinite(v) for v in (*previous, *target)):
            raise ValueError("Four finite previous and target speeds required")
        if stop:
            return (0.0,) * 4
        delta = self.wheel_acceleration_rps2 * dt
        return tuple(old + max(-delta, min(delta, new - old)) for old, new in zip(previous, target, strict=True))

    def advance_body(self, previous, target, dt, stop=False):
        if (
            not math.isfinite(dt)
            or dt <= 0
            or len(previous) != 3
            or len(target) != 3
            or not all(math.isfinite(v) for v in (*previous, *target))
        ):
            raise ValueError("Finite body velocities and positive dt required")
        if stop:
            return (0.0,) * 3
        return tuple(
            old + max(-a * dt, min(a * dt, new - old))
            for old, new, a in zip(previous, target, self.body_acceleration_limits, strict=True)
        )
