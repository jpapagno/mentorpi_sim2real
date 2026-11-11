import unittest
import numpy as np
from mentorpi_nav.robot.live_occupancy import live_occupancy_from_scan

class LiveOccupancyTests(unittest.TestCase):
    def test_wall_clears_visible_ray_blocks_hit_and_leaves_occlusion_unknown(self):
        points=np.column_stack((np.ones(51),np.linspace(-.5,.5,51)))
        grid=live_occupancy_from_scan(points)
        self.assertEqual(grid.shape,(100,100))
        self.assertFalse(grid[40,50])
        self.assertTrue(grid[30,50])
        self.assertTrue(grid[20,50])
        self.assertTrue(grid[50,10])
    def test_removed_obstacle_is_cleared_by_next_observed_farther_ray(self):
        near=live_occupancy_from_scan(np.array([[.5,0]]))
        far=live_occupancy_from_scan(np.array([[2.,0]]))
        self.assertTrue(near[40,50])
        self.assertFalse(far[40,50])
        self.assertTrue(far[10,50])
    def test_invalid_returns_do_not_certify_free_space(self):
        grid=live_occupancy_from_scan(np.array([[float('nan'),0],[float('inf'),0]]))
        self.assertTrue(grid[30,50])
        self.assertFalse(grid[49,49])
    def test_finite_return_beyond_grid_clears_ray_to_boundary(self):
        grid=live_occupancy_from_scan(np.array([[8.,0]]))
        self.assertFalse(grid[1,50])
        self.assertFalse(grid[0,50])
