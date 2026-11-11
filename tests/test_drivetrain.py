"""Measured channel convention, inverse/forward consistency and actuator bounds."""
import math
import random
import unittest
from dataclasses import replace
import torch
from mentorpi_nav.drivetrain import DriveConfig
from mentorpi_nav.core import NavigationCfg, NavigationState

class DrivetrainTests(unittest.TestCase):
    def test_measured_channels_and_independent_axis_patterns(self):
        drive = DriveConfig()
        c = math.pi*.065
        for axis, expected in [((.1,0,0),(-1,-1,1,1)),
                               ((0,.1,0),(1,-1,-1,1)),
                               ((0,0,.1),(1,1,1,1))]:
            wheels = drive.wheel_speeds(axis)
            self.assertEqual(tuple(1 if w>0 else -1 for w in wheels),expected)
        self.assertAlmostEqual(drive.wheel_speeds((.1,0,0))[0],-.1/c)
        # Explicitly distinguish rear-right channel 3 from front-right channel 4.
        self.assertLess(drive.wheel_speeds((0,.1,0))[2],0)
        self.assertGreater(drive.wheel_speeds((0,.1,0))[3],0)

    def test_round_trip_and_uniform_saturation(self):
        drive = DriveConfig()
        rng = random.Random(7)
        for _ in range(100):
            body = tuple(rng.uniform(-1,1) for _ in range(3))
            raw = drive.wheel_speeds(body,limit=False)
            reconstructed = drive.body_velocity(raw)
            for a,b in zip(body,reconstructed): self.assertAlmostEqual(a,b)
            limited = drive.wheel_speeds(body)
            self.assertLessEqual(max(map(abs,limited)),drive.max_wheel_rps+1e-12)
            ratio = max(1,max(map(abs,raw))/drive.max_wheel_rps)
            for a,b in zip(drive.body_velocity(limited),body): self.assertAlmostEqual(a,b/ratio)

    def test_stop_bypasses_ramp_and_invalid_commands_rejected(self):
        drive = DriveConfig()
        self.assertEqual(drive.advance((.6,)*4,(.8,)*4,.02,stop=True),(0.,)*4)
        self.assertEqual(drive.advance_body((.1,)*3,(0.,)*3,.02,stop=True),(0.,)*3)
        self.assertAlmostEqual(drive.advance((0.,)*4,(.8,)*4,.02)[0],.04)
        with self.assertRaises(ValueError):drive.wheel_speeds((float('nan'),0,0))
        with self.assertRaises(ValueError):DriveConfig(wheel_gains=(1,1))

    def test_simulation_matches_shared_scalar_motor_model(self):
        cfg = NavigationCfg(obstacle_count=0,drive_model='mecanum_v1',wheel_gains=(.9,1,.8,1))
        state = NavigationState(1,cfg)
        state.goal[:]=torch.tensor([1.,1.])
        drive = cfg.drivetrain_config()
        body,wheels = (0.,)*3,(0.,)*4
        actions = [(1.,.7,.4)]*12+[(0.,0.,0.)]*20
        for action in actions:
            target = tuple(a*b for a,b in zip(action,cfg.speed_limits))
            body = drive.advance_body(body,target,cfg.dt)
            command = drive.wheel_speeds(body)
            wheels = drive.advance(wheels,command,cfg.dt)
            expected = drive.body_velocity(drive.realized_wheels(wheels))
            state.step(torch.tensor([action]))
            torch.testing.assert_close(state.wheel_rps,torch.tensor([wheels]),atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(state.velocity,torch.tensor([expected]),atol=1e-6,rtol=1e-5)
        state.reset([0])
        self.assertEqual(float(state.wheel_rps.abs().sum()),0)
        self.assertEqual(float(state.body_command_velocity.abs().sum()),0)

    def test_deadband_does_not_prevent_startup(self):
        state = NavigationState(1,NavigationCfg(obstacle_count=0,drive_model='mecanum_v1',wheel_deadband_rps=.3))
        state.goal[:]=torch.tensor([1.,1.])
        for _ in range(10):state.step(torch.tensor([[1.,0.,0.]]))
        self.assertGreater(float(state.velocity[0,0]),.05)

    def test_response_compensation_is_uniform_and_still_wheel_bounded(self):
        nominal=DriveConfig(max_wheel_rps=1.2)
        boosted=replace(nominal,command_gain=1.5)
        target=(.05,.02,.1)
        for a,b in zip(boosted.wheel_speeds(target),nominal.wheel_speeds(target)):
            self.assertAlmostEqual(a,1.5*b)
        self.assertLessEqual(max(map(abs,boosted.wheel_speeds((.2,.2,.5)))),1.2)

if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
