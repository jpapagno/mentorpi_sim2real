import copy
import math
import unittest
import subprocess
import sys
import json
from mentorpi_nav.robot.motion_profile import (DIRECTIONS,VERSION,validate_profile,identity_profile,
    fit_profile,compensated_command,fixed_controller,predicted_rate,calibration_path_clear)


class MotionProfileTests(unittest.TestCase):
    def test_target_cli_previews_without_ros_or_hardware(self):
        r=subprocess.run([sys.executable,'-m','mentorpi_nav.robot.track_motion','--target','.06','0','0','--output','unused.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        report=json.loads(r.stdout)
        self.assertTrue(report['preview']);self.assertEqual(report['nonzero_commands_sent'],0)

    def test_target_cli_rejects_out_of_bounds_before_hardware_access(self):
        r=subprocess.run([sys.executable,'-m','mentorpi_nav.robot.track_motion','--target','.2','0','0','--output','unused.json','--execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        self.assertIn('limited to12cm',r.stderr)
    def test_calibration_clearance_rejects_unknown_and_blocked_path(self):
        self.assertFalse(calibration_path_clear([], [.08,0]))
        self.assertFalse(calibration_path_clear([[math.nan,0]]*20,[.08,0]))
        self.assertFalse(calibration_path_clear([[.22,0]]*20,[.08,0]))
        self.assertTrue(calibration_path_clear([[1.,0]]*20,[.08,0]))
    def test_identity_preserves_velocity_and_bounds(self):
        profile=validate_profile(identity_profile())
        self.assertEqual(compensated_command(profile,[.02,-.03,.12]),[.02,-.03,.12])
        self.assertEqual(compensated_command(profile,[1,-1,-2]),[.1,-.1,-.3])
        self.assertEqual(fixed_controller([0,0],0,profile),([0.,0.,0.],True))

    def test_fit_monotonic_directional_response(self):
        samples=[]
        for name,axis,_ in DIRECTIONS:
            for u,v in ([(.2,0.),(.25,.03),(.3,.06)] if axis==2 else [(.06,0.),(.07,.04),(.1,.08)]):
                samples.append({'direction':name,'command':u,'response':v})
        profile=fit_profile(samples)
        command=compensated_command(profile,[.001,-.001,-.001])
        self.assertEqual(command,[.07,-.07,-.25])
        self.assertGreater(compensated_command(profile,[.06,0,0])[0],.07)

    def test_same_controller_tracks_deadband_plant_with_changed_profile(self):
        profile=identity_profile()
        for name,axis,_ in DIRECTIONS:
            if axis<2:profile['directions'][name]={'knots':[[0,0],[.06,0],[.07,.07],[.1,.1]]}
        def rollout(calibration):
            position=0.
            for _ in range(200):
                command,done=fixed_controller([.08-position,0],0,calibration)
                if done:break
                # Independent synthetic plant: requested speed below0.065 stalls.
                velocity=command[0] if abs(command[0])>=.065 else 0.
                position+=velocity*.05
            return abs(.08-position)
        self.assertGreater(rollout(identity_profile()),.04)
        self.assertLessEqual(rollout(profile),.01)

    def test_invalid_profile_and_measurement_rejected(self):
        profile=identity_profile();profile['directions']['cw']['knots'][-1][0]=.8
        with self.assertRaises(ValueError):validate_profile(profile)
        with self.assertRaises(ValueError):fixed_controller([math.nan,0],0,identity_profile())
        with self.assertRaises(ValueError):fit_profile([])
