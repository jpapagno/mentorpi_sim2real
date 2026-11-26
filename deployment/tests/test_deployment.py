import math
import unittest

import numpy as np
from mentorpi_deploy import isaac_motion
from mentorpi_deploy.cli import parse_args
from mentorpi_deploy.command_guard import CommandGuard
from mentorpi_deploy.drivetrain import DriveConfig


class MotionContractTests(unittest.TestCase):
    def test_distance_interface_defaults_to_disarmed(self):
        args = parse_args(["1"])
        self.assertEqual(args.goal_relative, [1.0, 0.0])
        self.assertTrue(args.observe_only)
        self.assertEqual(args.policy, "isaaclab-motion-900")

    def test_driving_requires_explicit_flag_and_uses_bounds(self):
        args = parse_args(["1", "--drive"])
        self.assertFalse(args.observe_only)
        self.assertEqual(args.seconds, 20)
        self.assertAlmostEqual(args.max_displacement, 1.1)

    def test_observation_order_and_world_frame_rays(self):
        state = isaac_motion.observation(
            base=np.array([9.0, 22.0, math.pi]),
            start=np.array([10.0, 20.0, math.pi / 2]),
            body_velocity=np.array([0.1, 0.2]),
            goal=np.array([7.0, 25.0]),
            body_scan_points=np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.5]]),
        )
        self.assertEqual(state.shape, (1, 80))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state[0, :8], [2.0, 1.0, 0.04, -0.2, 0.1, 0.0, 3.0, 2.0])
        ranges = np.full(72, 3.0)
        ranges[0] = math.hypot(0.5, 0.06)
        ranges[54] = math.hypot(1.0, 0.06)
        np.testing.assert_allclose(state[0, 8:], ranges)

    def test_issued_velocity_uses_body_axes(self):
        np.testing.assert_allclose(
            isaac_motion.body_command(np.array([0.08, 0.0]), math.pi / 2), [0.0, -0.08, 0.0], atol=1e-9
        )

    def test_goal_uses_the_training_square(self):
        self.assertTrue(isaac_motion.goal_reached(np.array([0.14, 0.14]), 0.0))
        self.assertFalse(isaac_motion.goal_reached(np.array([0.16, 0.0]), 0.0))


class MotorPathTests(unittest.TestCase):
    def test_forward_wheel_signs_and_slew_bypass(self):
        drive = DriveConfig()
        wheels = drive.wheel_speeds((0.08, 0.0, 0.0))
        self.assertEqual(tuple(math.copysign(1, wheel) for wheel in wheels), (-1, -1, 1, 1))
        self.assertEqual(drive.advance(wheels, wheels, 0.02, stop=True), (0.0,) * 4)

    def test_guard_drops_old_commands_on_arm_and_expiry(self):
        guard = CommandGuard()
        guard.scan_at = guard.imu_at = 1.0
        guard.receive_command((0.2, 0.0, 0.0), 1.0)
        guard.receive_enable(True, 1.01)
        self.assertEqual(guard.output(1.01)[0], (0.0,) * 3)
        guard.receive_command((0.2, 0.0, 0.0), 1.02)
        self.assertEqual(guard.output(1.03), ((0.1, 0.0, 0.0), []))
        self.assertEqual(guard.output(1.23), ((0.0,) * 3, ["command_stale"]))
        self.assertEqual(guard.output(1.41)[0], (0.0,) * 3)
        self.assertIn("scan_stale", guard.output(1.41)[1])


if __name__ == "__main__":
    unittest.main()
