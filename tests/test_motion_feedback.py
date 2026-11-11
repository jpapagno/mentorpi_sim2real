import math
import unittest
from deployment.runtime.motion_feedback import PlanarVelocityEstimator,MotionProgressMonitor

class MotionFeedbackTests(unittest.TestCase):
    def test_measured_sideways_velocity_rotates_into_body_frame(self):
        estimator=PlanarVelocityEstimator()
        for i in range(5):velocity=estimator.add(i*.1,-.1*i*.1,0.,math.pi/2)
        self.assertAlmostEqual(velocity[0],0.,places=7)
        self.assertAlmostEqual(velocity[1],.1,places=7)
        self.assertAlmostEqual(velocity[2],0.,places=7)
    def test_zero_motion_does_not_become_command_velocity(self):
        estimator=PlanarVelocityEstimator()
        for i in range(5):velocity=estimator.add(i*.1,1.,2.,0.)
        self.assertEqual(velocity,(0.,0.,0.))
    def test_progress_monitor_stops_stall_but_allows_translation_and_rotation(self):
        monitor=MotionProgressMonitor()
        self.assertIsNone(monitor.update(0,[0,0,0],[.06,0,0]))
        self.assertEqual(monitor.update(3,[0,0,0],[.06,0,0]),'commanded_without_measured_motion')
        for pose in ([.05,0,0],[0,.05,0],[0,0,.3]):
            monitor=MotionProgressMonitor();monitor.update(0,[0,0,0],[.06,0,.3])
            self.assertIsNone(monitor.update(3,pose,[.06,0,.3]))
    def test_wraparound_and_discontinuities(self):
        estimator=PlanarVelocityEstimator()
        for i in range(5):velocity=estimator.add(i*.1,0.,0.,(3.12+.1*i*.1+math.pi)%(2*math.pi)-math.pi)
        self.assertAlmostEqual(velocity[2],.1,places=6)
        self.assertIsNone(estimator.add(2,5,5,0))
