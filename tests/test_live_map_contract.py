import math
import unittest
import numpy as np
from mentorpi_nav.robot.navigation_math import state_vector

class PolicyStateContractTests(unittest.TestCase):
    def test_goal_velocity_and_previous_action_are_in_body_frame(self):
        state=state_vector([1.,0.],[0.,0.,math.pi/2],[.1,0.,.25],[.2,-.1,0.])
        self.assertAlmostEqual(float(state[0]),0.,places=6)
        self.assertAlmostEqual(float(state[1]),-1/(4*math.sqrt(2)),places=6)
        np.testing.assert_allclose(state[2:5],[.5,0.,.5])
        np.testing.assert_allclose(state[5:],[.2,-.1,0.])
