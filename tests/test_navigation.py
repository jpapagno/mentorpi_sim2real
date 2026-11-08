"""Behaviour checks for the policy interface, swept collision, and reset sampler."""

from dataclasses import replace
import math
import random
import unittest

import torch

from mentorpi_nav.core import NavigationCfg, NavigationState, sample_layout, swept_collision, world_to_body


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = NavigationCfg(obstacle_count=0)
        self.state = NavigationState(2, self.cfg)
        self.state.reset()
        self.state.xy[:] = 0
        self.state.yaw[:] = 0
        self.state.goal[:] = torch.tensor([1.0, 1.0])

    def test_body_frame_rotation(self):
        actual = world_to_body(torch.tensor([[1.0, 0.0]]), torch.tensor([math.pi / 2]))
        torch.testing.assert_close(actual, torch.tensor([[0.0, -1.0]]), atol=1e-6, rtol=0)

    def test_strafe_and_rotated_forward(self):
        self.state.yaw[1] = math.pi / 2
        self.state.step(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        self.assertTrue((self.state.xy[:, 1] > 0).all())
        self.assertTrue((self.state.xy[:, 0].abs() < 1e-6).all())

    def test_acceleration_and_speed_limits(self):
        for _ in range(10):
            before = self.state.velocity.clone()
            self.state.step(torch.ones(2, 3) * 10)
            self.assertTrue(((self.state.velocity - before).abs() <= self.state.acceleration * self.cfg.dt + 1e-6).all())
            self.assertTrue((self.state.velocity.abs() <= self.state.limits).all())
        for _ in range(10):
            self.state.step(torch.zeros(2, 3))
        torch.testing.assert_close(self.state.velocity, torch.zeros(2, 3))

    def test_swept_collision_catches_tunnelling(self):
        start, end = torch.tensor([[-1.0, 0.0]]), torch.tensor([[1.0, 0.0]])
        obs = torch.tensor([[[0.0, 0.0, 0.1]]])
        self.assertTrue(swept_collision(start, end, obs, torch.tensor([[True]]), 0.15, 2.0).item())
        self.assertFalse(swept_collision(start, end, obs, torch.tensor([[False]]), 0.15, 2.0).item())
        self.assertFalse(swept_collision(start + 0.5, end + 0.5, obs, torch.tensor([[True]]), 0.15, 2.0).item())

    def test_tangent_and_stationary_overlap_are_collisions(self):
        obs = torch.tensor([[[0.0, 0.0, 0.1]]])
        valid = torch.tensor([[True]])
        self.assertTrue(swept_collision(torch.tensor([[-1.0, 0.25]]), torch.tensor([[1.0, 0.25]]), obs, valid, 0.15, 2.0))
        self.assertTrue(swept_collision(torch.zeros(1, 2), torch.zeros(1, 2), obs, valid, 0.15, 2.0))

    def test_wall_collision_keeps_robot_inside(self):
        self.state.xy[0] = torch.tensor([1.8499, 0.0])
        old = self.state.xy[0].clone()
        self.state.step(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        self.assertTrue(self.state.collision[0])
        torch.testing.assert_close(self.state.xy[0], old)
        self.assertFalse(self.state.collision[1])

    def test_grid_shape_binary_values_and_separate_state(self):
        obs = self.state.observations()
        self.assertEqual(set(obs), {"policy", "occupancy"})
        self.assertEqual(obs["policy"].shape, (2, 8))
        self.assertEqual(obs["occupancy"].shape, (2, 1, 100, 100))
        self.assertTrue(((obs["occupancy"] == 0) | (obs["occupancy"] == 1)).all())
        self.assertEqual(obs["occupancy"][0, 0, 49:51, 49:51].sum(), 0)

    def test_grid_marks_correct_cell_and_ignores_invalid_obstacles(self):
        self.state.obstacles[0, 0] = torch.tensor([1.225, -0.025, 0.02])
        before = self.state.occupancy_grid()
        self.state.valid[0, 0] = True
        after = self.state.occupancy_grid()
        difference = (after - before).nonzero().tolist()
        self.assertEqual(difference, [[0, 0, 25, 50]])
        torch.testing.assert_close(after[1], before[1])

    def test_grid_rotates_when_robot_turns(self):
        self.state.obstacles[:, 0] = torch.tensor([1.225, -0.025, 0.08])
        self.state.valid[:, 0] = True
        self.state.yaw[1] = math.pi / 2
        grid = self.state.occupancy_grid()
        torch.testing.assert_close(grid[1, 0], torch.rot90(grid[0, 0], -1, (0, 1)))

    def test_grid_moves_with_robot(self):
        self.state.obstacles[:, 0] = torch.tensor([1.225, -0.025, 0.02])
        self.state.valid[:, 0] = True
        self.state.xy[1, 0] = 0.5
        grid = self.state.occupancy_grid()
        self.assertEqual(grid[0, 0, 25, 50], 1)
        self.assertEqual(grid[1, 0, 35, 50], 1)
        self.assertEqual(grid[1, 0, 25, 50], 0)

    def test_grid_wall_tiles_and_outside_are_blocked(self):
        grid = self.state.occupancy_grid()[0, 0]
        self.assertTrue((grid[:10] == 1).all())
        self.assertTrue((grid[:, :10] == 1).all())
        self.state.xy[0, 0] = 1.8
        grid = self.state.occupancy_grid()[0, 0]
        self.assertEqual(grid[40, 50], 1)  # ahead of the wall
        self.assertEqual(grid[49, 50], 0)  # immediately ahead of the robot

    def test_thin_obstacle_on_cell_boundary_is_not_lost(self):
        self.state.obstacles[0, 0] = torch.tensor([1.0, 0.0, 0.001])
        self.state.valid[0, 0] = True
        grid = self.state.occupancy_grid()[0, 0]
        self.assertTrue((grid[29:31, 49:51] == 1).all())

    def test_optional_inflation_is_binary_and_expands_blocked_area(self):
        self.state.obstacles[0, 0] = torch.tensor([1.0, 0.0, 0.1])
        self.state.valid[0, 0] = True
        raw = self.state.occupancy_grid()
        self.state.cfg = replace(self.cfg, occupancy_inflation=0.15)
        inflated = self.state.occupancy_grid()
        self.assertTrue((inflated >= raw).all())
        self.assertGreater(inflated.sum(), raw.sum())

    def test_reset_is_partial_and_clears_history(self):
        self.state.velocity[:] = 1
        self.state.actions[:] = 1
        unchanged = self.state.xy[1].clone()
        self.state.reset(torch.tensor([0], dtype=torch.int32))
        torch.testing.assert_close(self.state.velocity[0], torch.zeros(3))
        torch.testing.assert_close(self.state.actions[0], torch.zeros(3))
        torch.testing.assert_close(self.state.xy[1], unchanged)
        self.assertTrue((self.state.velocity[1] == 1).all())

    def test_progress_reward_and_success(self):
        self.state.goal[:] = torch.tensor([1.0, 0.0])
        self.state.step(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
        self.assertGreater(self.state.reward[0], self.state.reward[1])
        self.state.xy[0] = self.state.goal[0]
        self.state.velocity[:] = 0
        self.state.step(torch.zeros(2, 3))
        self.assertTrue(self.state.success[0])

    def test_sampled_starts_goals_clear_and_separated(self):
        cfg = replace(self.cfg, obstacle_count=8)
        rng = random.Random(7)
        for _ in range(30):
            start, goal, obs, yaw = sample_layout(cfg, rng)
            self.assertGreaterEqual(math.dist(start, goal), 1.0)
            for point in (start, goal):
                self.assertLess(max(abs(x) for x in point), cfg.arena_half_extent - cfg.robot_radius)
                for x, y, r in obs:
                    self.assertGreater(math.dist(point, (x, y)), r + cfg.robot_radius)

    def test_reproducible_layouts(self):
        a, b = random.Random(123), random.Random(123)
        self.assertEqual(sample_layout(self.cfg, a), sample_layout(self.cfg, b))

    def test_invalid_actions_and_config(self):
        with self.assertRaises(ValueError):
            self.state.step(torch.full((2, 3), float("nan")))
        with self.assertRaises(ValueError):
            NavigationCfg(obstacle_count=9)


if __name__ == "__main__":
    unittest.main()
