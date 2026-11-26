"""Measure bounded motor pulses against RF2O odometry with the deployed guard."""

import argparse
import json
import math
import time
from pathlib import Path

from .command_guard import BODY_LIMITS

DIRECTIONS = {"forward": (0, 1), "backward": (0, -1), "left": (1, 1), "right": (1, -1), "ccw": (2, 1), "cw": (2, -1)}
ROBOT_RADIUS_M = math.hypot(0.254, 0.2032) / 2 + 0.02


def body_delta(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return [c * dx + s * dy, -s * dx + c * dy, (end[2] - start[2] + math.pi) % (2 * math.pi) - math.pi]


def collect(node, args, result):
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry

    from .frames import yaw_of

    pose = None
    measured_at = float("-inf")

    def receive(msg):
        nonlocal pose, measured_at
        if -0.05 <= node.age(msg) < 0.4:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pose = [p.x, p.y, yaw_of([q.x, q.y, q.z, q.w])]
            measured_at = time.monotonic()

    node.create_subscription(Odometry, "/odom_rf2o", receive, 10)
    ready_by = time.monotonic() + 12
    while time.monotonic() < ready_by:
        node.phase(0.05, (0.0, 0.0, 0.0), False)
        if not node.health() and time.monotonic() - measured_at < 0.4 and node.status:
            break
    else:
        raise RuntimeError(f"Calibration sensors or guard unavailable. Health faults {node.health()}")
    session_origin = pose

    def pulse(command, seconds, origin, enabled):
        deadline = time.monotonic() + seconds
        samples = []
        while (now := time.monotonic()) < deadline:
            health = node.health()
            if health or now - measured_at >= 0.4:
                raise RuntimeError(f"Calibration feedback stale. Health faults {health}")
            measured = body_delta(origin, pose)
            if math.hypot(*measured[:2]) > (0.12 if enabled else 0.05) or abs(measured[2]) > (0.5 if enabled else 0.12):
                raise RuntimeError("Calibration pulse or settling displacement exceeded limit")
            total = body_delta(session_origin, pose)
            if math.hypot(*total[:2]) > 0.3 or abs(total[2]) > 1.0:
                raise RuntimeError("Calibration session displacement exceeded limit")
            if enabled and args.check_clearance:
                ranges = np.linalg.norm(node.scan_points()[:, :2], axis=1)
                if len(ranges) < 20 or ranges.min() <= ROBOT_RADIUS_M:
                    raise RuntimeError("Calibration clearance check failed")
            node.publish(command, enabled)
            samples.append({"time_s": now, "laser_pose": pose, "guard": node.status})
            until = min(deadline, now + 0.05)
            while (remaining := until - time.monotonic()) > 0:
                rclpy.spin_once(node, timeout_sec=min(remaining, 0.005))
        return samples

    for repeat in range(args.repetitions):
        for direction in args.directions:
            axis, sign = DIRECTIONS[direction]
            command = [0.0, 0.0, 0.0]
            command[axis] = sign * (args.yaw_speed if axis == 2 else args.linear_speed)
            start = pose
            row = {"direction": direction, "repeat": repeat + 1, "command": command}
            result["pulses"].append(row)
            row["pulse_samples"] = pulse(command, args.pulse_seconds, start, True)
            row["settling_samples"] = pulse((0.0, 0.0, 0.0), 1.0, pose, False)
            delta = body_delta(start, pose)
            row["measured_delta"] = delta
            applied = [sample["guard"].get("applied", [0.0, 0.0, 0.0])[axis] for sample in row["pulse_samples"]]
            if max((abs(value) for value in applied), default=0.0) < 0.005:
                raise RuntimeError("Motor command was blocked")
            if sign * delta[axis] < -0.003:
                raise RuntimeError("Measured motion opposes the requested axis")
            if abs(delta[axis]) < (0.02 if axis == 2 else 0.01):
                raise RuntimeError("Insufficient measured response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send bounded physical motor commands")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=["forward", "backward", "left", "right"])
    parser.add_argument("--linear-speed", type=float, default=0.06)
    parser.add_argument("--yaw-speed", type=float, default=0.2)
    parser.add_argument("--pulse-seconds", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--check-clearance", action="store_true", help="Stop if the scan intersects the robot clearance circle"
    )
    args = parser.parse_args()
    if not (
        0 < args.linear_speed <= BODY_LIMITS[0]
        and 0 < args.yaw_speed <= BODY_LIMITS[2]
        and 0 < args.pulse_seconds <= 1.5
        and args.linear_speed * args.pulse_seconds <= 0.100001
    ):
        parser.error("Pulse exceeds calibration limits")
    settings = {key: value for key, value in vars(args).items() if key != "output"}
    if not args.execute:
        print(json.dumps(settings, indent=2))
        return 0

    from .policy_io import control_session

    result = {"settings": settings, "pulses": [], "passed": False}
    try:
        with control_session() as node:
            collect(node, args, result)
        result["passed"] = True
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = str(exc) or type(exc).__name__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Calibration {'passed' if result['passed'] else 'stopped'}. Trace written to {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
