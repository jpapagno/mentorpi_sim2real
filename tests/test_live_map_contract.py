import math
import unittest
import numpy as np
from deployment.runtime.navigation_math import occupancy_from_map,overlay_points,state_vector,subgoal_on_path


class LiveMapContractTests(unittest.TestCase):
    def test_forward_left_and_rotation(self):
        source=np.zeros((200,200),dtype=np.int16)
        source[100:105,120:125]=100
        grid=occupancy_from_map(source,.05,[-5.,-5.,0.],[0.,0.,0.])
        self.assertTrue(grid[25:35,40:55].any())
        rotated=occupancy_from_map(source,.05,[-5.,-5.,0.],[0.,0.,math.pi/2])
        self.assertTrue(rotated[40:55,65:75].any())

    def test_unknown_blocks_without_footprint_inflation(self):
        source=np.full((200,200),-1,dtype=np.int16)
        self.assertTrue(occupancy_from_map(source,.05,[-5.,-5.,0.],[0.,0.,0.]).all())
        source[:]=0
        grid=occupancy_from_map(source,.05,[-5.,-5.,0.],[0.,0.,0.])
        self.assertFalse(grid.any())
        grid=overlay_points(grid,np.array([[1.,.5]]),uncertainty_cells=0)
        self.assertTrue(grid[30,40])
        self.assertEqual(int(grid.sum()),1)

    def test_goal_state_and_path_subgoal(self):
        state=state_vector([1.,0.],[0.,0.,math.pi/2],[.1,0.,.25],[.2,-.1,0.])
        self.assertAlmostEqual(float(state[0]),0.,places=6)
        self.assertAlmostEqual(float(state[1]),-1/(4*math.sqrt(2)),places=6)
        np.testing.assert_allclose(state[2:5],[.5,0.,.5])
        target=subgoal_on_path([[0.,0.],[.5,0.],[1.,0.],[1.5,0.]],[.1,0.],1.)
        np.testing.assert_allclose(target,[1.,0.])
