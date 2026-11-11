import math
import unittest
from deployment.runtime.turn_control import turn_decision

class TurnControlTests(unittest.TestCase):
    def test_signed_targets_and_slowdown(self):
        self.assertEqual(turn_decision(math.pi/4,0,0,0,0,7400),('turning',.85))
        self.assertEqual(turn_decision(-math.pi/4,0,0,0,0,7400),('turning',-.85))
        self.assertEqual(turn_decision(math.pi/4,math.radians(40),1,1,0,7400),('turning',.5))
        self.assertEqual(turn_decision(math.pi/4,math.radians(44),1,1,0,7400),('target_reached',0.))

    def test_faults_never_command_turning(self):
        cases=[(math.pi/4,0,4,0,0,7400),(math.pi/4,0,0,math.pi/2,0,7400),
               (math.pi/4,0,0,0,.3,7400),(math.pi/4,0,0,0,0,6300),
               (math.pi/4,math.radians(-6),1,0,0,7400),
               (math.pi/4,math.radians(56),1,0,0,7400)]
        for args in cases:
            status,speed=turn_decision(*args)
            self.assertTrue(status.startswith('fault:'))
            self.assertEqual(speed,0.)
