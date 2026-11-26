"""Start a local-goal policy run in the configured ROS container."""

import argparse
import math
import os
import shlex
import subprocess

from .goals import add_arguments, validate
from .paths import ROOT
from .policy_selection import resolve_policy

CONTAINER_ROOT = "/home/ubuntu/shared/mentorpi_isaac_deploy"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("distance", type=float, nargs="?", help="Goal distance straight ahead in metres")
    parser.add_argument("--drive", action="store_true", help="Arm the motors for a distance command")
    parser.add_argument("--policy", default="isaaclab-motion-900", help="Bundle directory name under models/")
    parser.add_argument("--stop", action="store_true", help="Stop navigation and motor output")
    args = parser.parse_args(argv)
    if args.stop:
        return args
    if args.drive and (args.distance is None or args.observe_only):
        parser.error("--drive requires a distance and cannot be combined with --observe-only")
    if args.distance is not None:
        if args.goal_relative is not None or args.goal_odom is not None:
            parser.error("Supply a distance or an explicit goal")
        if not math.isfinite(args.distance) or args.distance <= 0:
            parser.error("Distance must be finite and positive")
        args.goal_relative = [args.distance, 0.0]
        args.observe_only = not args.drive
        if args.drive:
            if args.seconds is None:
                args.seconds = 20.0
            if args.max_displacement is None:
                args.max_displacement = args.distance + 0.1
    validate(parser, args)
    return args


def main():
    args = parse_args()
    if not args.stop:
        resolve_policy(ROOT, args.policy)
    if getattr(os, "geteuid", lambda: -1)() != 0:
        raise SystemExit("Run with sudo python3 on the Pi")
    service = ["systemctl", "stop" if args.stop else "start", "mentorpi-navigation.service"]
    subprocess.run(service, check=True)
    if args.stop:
        return 0
    command = ["python3", "-m", "mentorpi_deploy.policy", "--policy", args.policy]
    if args.goal_relative is not None:
        command += ["--goal-relative", *map(str, args.goal_relative)]
    else:
        command += ["--goal-odom", *map(str, args.goal_odom)]
    if args.seconds is not None:
        command += ["--seconds", str(args.seconds)]
    if args.max_displacement is not None:
        command += ["--max-displacement", str(args.max_displacement)]
    if args.observe_only:
        command += ["--observe-only"]
    print(
        "Running disarmed inference."
        if args.observe_only
        else "Starting policy. Goal coordinates are metres. Ctrl+C stops the robot.",
        flush=True,
    )
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "ubuntu",
                "-w",
                CONTAINER_ROOT,
                "MentorPi",
                "/bin/zsh",
                "-c",
                "source /home/ubuntu/.zshrc >/dev/null; exec " + shlex.join(command),
            ]
        )
        return result.returncode
    except KeyboardInterrupt:
        subprocess.run(["systemctl", "stop", "mentorpi-navigation.service"])
        print("Navigation stopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
