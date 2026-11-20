import math
import unittest
from mentorpi_nav.robot.calibrate_motion import body_delta,measurable_response,displacement_exceeded


class MotionCalibrationTests(unittest.TestCase):
    def test_large_post_stop_motion_aborts_before_next_pulse(self):
        self.assertTrue(displacement_exceeded([0,0,-.26],settling=True))
        self.assertTrue(displacement_exceeded([.051,0,0],settling=True))
        self.assertFalse(displacement_exceeded([.002,0,.07],settling=True))
    def test_body_coordinates_and_yaw_wrap(self):
        delta=body_delta([0,0,math.pi/2],[0,1,math.pi/2])
        self.assertAlmostEqual(delta[0],1);self.assertAlmostEqual(delta[1],0)
        self.assertAlmostEqual(body_delta([0,0,math.pi-.01],[0,0,-math.pi+.01])[2],.02)

    def test_stationary_noise_is_not_calibration(self):
        self.assertFalse(measurable_response(0,[.001,.004,.003]))
        self.assertFalse(measurable_response(1,[.001,.004,.003]))
        self.assertFalse(measurable_response(2,[.001,.004,.003]))

    def test_both_directions_can_be_measured(self):
        self.assertTrue(measurable_response(0,[-.06,0,0]))
        self.assertTrue(measurable_response(1,[0,.06,0]))
        self.assertTrue(measurable_response(2,[0,0,-.2]))

    def test_numpy_response_serializes(self):
        import numpy as np
        import json
        value=measurable_response(0,np.array([.05,0,0]))
        self.assertIsInstance(value,bool)
        self.assertEqual(json.dumps({'response':value}),'{"response": true}')
