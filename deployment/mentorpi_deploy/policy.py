"""Run one policy with live ROS feedback, or verify its export offline."""

import argparse
import json
import math
import time

import numpy as np
import torch

from .frames import yaw_of
from .goals import add_arguments, validate
from .motion_feedback import MotionProgressMonitor
from .paths import ROOT
from .policy_runtime import PolicyRuntime
from .policy_selection import resolve_policy


def run(node, runtime, args, result):
    import rclpy
    from nav_msgs.msg import Odometry

    laser_pose = None
    laser_at = float("-inf")

    def receive_laser(msg):
        nonlocal laser_pose, laser_at
        if not -0.05 <= node.age(msg) < 0.4:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        laser_pose = np.array([p.x, p.y, yaw_of([q.x, q.y, q.z, q.w])])
        laser_at = time.monotonic()

    node.create_subscription(Odometry, "/odom_rf2o", receive_laser, 10)
    ready_by = time.monotonic() + 10
    while time.monotonic() < ready_by:
        node.phase(0.05, (0.0, 0.0, 0.0), False)
        if not node.health() and time.monotonic() - laser_at < 0.4 and node.status:
            break
    else:
        raise RuntimeError(f"Sensors or guard unavailable. Health faults {node.health()}")

    start, _ = node.pose()
    laser_start = laser_pose[:2].copy()
    if args.goal_odom is not None:
        goal = np.asarray(args.goal_odom)
    else:
        forward, left = args.goal_relative
        c, s = math.cos(start[2]), math.sin(start[2])
        goal = start[:2] + np.array([c * forward - s * left, s * forward + c * left])
    result.update(start=start.tolist(), goal=goal.tolist())
    monitor = MotionProgressMonitor()
    started = time.monotonic()
    velocity = np.zeros(3)
    ramp_step = np.array([0.2, 0.2, 0.5]) * runtime.period
    trace = result["trace"]
    scans = result["scans"]
    last_scan = None
    while True:
        tick = time.monotonic()
        if args.seconds is not None and tick - started >= args.seconds:
            return "time_limit"
        health = node.health()
        if health or tick - laser_at >= 0.4:
            raise RuntimeError(f"Sensor feedback missing or stale. Health faults {health}")
        base, _ = node.pose()
        displacement = max(np.linalg.norm(base[:2] - start[:2]), np.linalg.norm(laser_pose[:2] - laser_start))
        if args.max_displacement is not None and displacement >= args.max_displacement:
            return "displacement_limit"
        if runtime.adapter.goal_reached(goal - base[:2], start[2]):
            return "goal_reached"
        fault = monitor.update(tick, laser_pose, node.status.get("applied", (0.0, 0.0, 0.0)))
        if fault:
            raise RuntimeError(fault)

        measured = node.measured_odom.twist.twist.linear
        scan = node.scan
        points = node.scan_points()
        observation, raw_action, action, desired, inference_ms = runtime.step(
            base, start, [measured.x, measured.y], goal, points
        )
        rclpy.spin_once(node, timeout_sec=0.0)
        now = time.monotonic()
        health = node.health()
        if health or now - laser_at >= 0.4:
            raise RuntimeError(f"Feedback stale after inference. Health faults {health}")
        if args.seconds is not None and now - started >= args.seconds:
            return "time_limit"
        if not args.observe_only:
            velocity += np.clip(desired - velocity, -ramp_step, ramp_step)
        node.publish(velocity, not args.observe_only)
        if scan is not last_scan:
            stamp = scan.header.stamp
            scans.append(
                {
                    "stamp_ns": str(stamp.sec * 1_000_000_000 + stamp.nanosec),
                    "frame": scan.header.frame_id,
                    "angle_min": scan.angle_min,
                    "angle_increment": scan.angle_increment,
                    "range_min": scan.range_min,
                    "range_max": scan.range_max,
                    "ranges_m": [value if math.isfinite(value) else None for value in scan.ranges],
                    "body_points_m": points[:, :2].tolist(),
                }
            )
            last_scan = scan
        trace.append(
            {
                "elapsed_s": now - started,
                "pose": base.tolist(),
                "displacement_m": float(displacement),
                "policy_action": action.tolist(),
                "actor_output": raw_action.tolist(),
                "observation": observation.tolist(),
                "scan_index": len(scans) - 1,
                "desired_body_velocity": desired.tolist(),
                "requested_velocity": velocity.tolist(),
                "guard": node.status,
                "inference_ms": inference_ms,
                "scan_clearance_m": float(np.linalg.norm(points[:, :2], axis=1).min()),
            }
        )
        deadline = tick + runtime.period
        while (remaining := deadline - time.monotonic()) > 0:
            rclpy.spin_once(node, timeout_sec=min(remaining, 0.005))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--policy", required=True, help="Bundle directory name under models/")
    parser.add_argument("--verify-only", action="store_true", help="Check the bundle without ROS or motor access")
    args = parser.parse_args()
    if not args.verify_only:
        validate(parser, args)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = PolicyRuntime(resolve_policy(ROOT, args.policy))
    verification = runtime.verify()
    print(json.dumps({"policy": args.policy, **verification}), flush=True)
    if args.verify_only:
        return 0

    from .policy_io import control_session

    diagnostics = ROOT / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    timestamp = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns()}"
    output = diagnostics / f"policy-{timestamp}.json"
    result = {
        "episode_version": 1,
        "policy": args.policy,
        "observe_only": args.observe_only,
        "verification": verification,
        "scans": [],
        "trace": [],
    }
    try:
        with control_session() as node:
            result["stop_reason"] = run(node, runtime, args, result)
    except KeyboardInterrupt:
        result["stop_reason"] = "interrupted"
    except Exception as exc:
        result.update(stop_reason="fault", error=str(exc))
    result["controls"] = len(result["trace"])
    result["passed"] = result["stop_reason"] == "goal_reached" or (
        args.observe_only and result["stop_reason"] == "time_limit" and result["controls"] > 0
    )
    output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"trace", "scans"}}), flush=True)
    print(f"Trace written to {output}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
