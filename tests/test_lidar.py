"""Visibility, runtime raster parity, and sensor timing for simulated LiDAR."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from mentorpi_nav.core import NavigationCfg, NavigationState
from mentorpi_nav.lidar import scan_grid
from mentorpi_nav.robot.live_occupancy import live_occupancy_from_scan
from mentorpi_nav.robot.scan_math import fixed_circle_scan


class LidarTests(unittest.TestCase):
    def state(self, batch=1, **kwargs):
        cfg = NavigationCfg(occupancy_source='lidar', obstacle_count=0, **kwargs)
        state = NavigationState(batch, cfg, seed=71)
        state.goal[:] = torch.tensor([1., 0.])
        return state

    def test_raster_matches_robot_with_missing_and_beyond_grid_returns(self):
        rng = np.random.default_rng(83)
        angle = np.linspace(-math.pi, math.pi, 720, endpoint=False)
        radius = rng.uniform(.2, 8., 720)
        origin = np.array([-.012242, -.00008533])
        points = np.column_stack((radius*np.cos(angle), radius*np.sin(angle))) + origin
        points[::13] = np.nan
        points[1::71] = np.inf
        actual = scan_grid(torch.tensor(points[None]), torch.tensor(origin))[0, 0].numpy()
        np.testing.assert_array_equal(actual, live_occupancy_from_scan(points, origin))
        empty = np.full((1, 720, 2), np.nan)
        np.testing.assert_array_equal(scan_grid(torch.tensor(empty), torch.tensor(origin))[0, 0],
                                      live_occupancy_from_scan(empty[0], origin))

    def test_first_hit_occludes_geometry_behind_it(self):
        state = self.state()
        state.obstacles[0, 0] = torch.tensor([.8, 0., .2])
        state.valid[0, 0] = True
        grid = state.occupancy_grid()
        self.assertAlmostEqual(float(state.lidar.ranges[0, 0]), .612242, places=6)
        self.assertEqual(float(grid[0, 0, 44, 50]), 0)  # visible before hit
        self.assertEqual(float(grid[0, 0, 20, 50]), 1)  # unknown behind obstacle
        geometry = NavigationState(1, replace(state.cfg, occupancy_source='geometry'))
        geometry.obstacles[:] = state.obstacles
        geometry.valid[:] = state.valid
        self.assertEqual(float(geometry.occupancy_grid()[0, 0, 20, 50]), 0)

    def test_rotation_mounting_and_raw_normalization_match_runtime(self):
        state = self.state(batch=2)
        state.xy[:] = torch.tensor([[.2, -.3], [-.4, .1]])
        state.yaw[:] = torch.tensor([0., math.pi / 2])
        actual = state.occupancy_grid().numpy()
        for i in range(2):
            ranges = state.lidar.ranges[i].numpy()
            normalized = fixed_circle_scan(ranges, 0., 2*math.pi/499, .02, 25.)
            # Normalizer rounds to float32; avoid placing fixture on cell edges.
            angle = -math.pi + np.arange(720)*2*math.pi/720
            valid = np.isfinite(normalized)
            points = np.column_stack((normalized[valid]*np.cos(angle[valid]),
                                      normalized[valid]*np.sin(angle[valid]))) + np.array(state.cfg.lidar_origin)
            expected = live_occupancy_from_scan(points, state.cfg.lidar_origin)
            np.testing.assert_array_equal(actual[i, 0], expected)
        # Front wall in body frame after a quarter turn is world +y.
        self.assertAlmostEqual(float(state.lidar.ranges[1, 0]), 1.912242, places=6)

    def test_scan_hold_refresh_and_partial_reset(self):
        state = self.state(batch=2)
        first = state.occupancy_grid()
        state.obstacles[:, 0] = torch.tensor([.7, 0., .2])
        state.valid[:, 0] = True
        state.step(torch.zeros(2, 3))
        torch.testing.assert_close(state.occupancy_grid(), first)
        state.step(torch.zeros(2, 3))
        second = state.occupancy_grid()
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(second, state.occupancy_grid())
        state.reset([0])
        torch.testing.assert_close(state.occupancy_grid()[1], second[1])

    def test_seeded_noise_and_dropout_are_repeatable_and_all_missing_is_unknown(self):
        a = self.state(lidar_noise_std=.002, lidar_dropout=.06)
        b = self.state(lidar_noise_std=.002, lidar_dropout=.06)
        torch.testing.assert_close(a.occupancy_grid(), b.occupancy_grid())
        torch.testing.assert_close(a.lidar.ranges, b.lidar.ranges)
        empty = self.state(lidar_dropout=1.)
        grid = empty.occupancy_grid()
        self.assertEqual(float(grid[0, 0, 30, 50]), 1)
        self.assertEqual(float(grid[0, 0, 49, 49]), 0)
        outside = self.state(lidar_range_max=.1)
        torch.testing.assert_close(outside.occupancy_grid(), grid)

    def test_invalid_sensor_contract_is_rejected(self):
        for values in ({'occupancy_resolution': .1}, {'occupancy_inflation': .1},
                       {'lidar_dropout': -1}, {'lidar_noise_std': float('nan')},
                       {'lidar_scan_period': .075}, {'lidar_rays': 1}, {'lidar_origin': (1., 0.)},
                       {'lidar_noise_std_range': (.01, .001)}, {'lidar_dropout_range': (0., 1.1)},
                       {'lidar_dropout_sectors': -1}, {'lidar_dropout_sector_degrees': (0., 5.)},
                       {'lidar_outlier_probability': 1.1}, {'lidar_outlier_error': -1.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.state(**values)

    def test_episode_sensor_randomization_and_partial_reset_isolation(self):
        settings = dict(lidar_noise_std_range=(.003, .012), lidar_dropout_range=(.03, .1),
                        lidar_dropout_sectors=3, lidar_dropout_sector_degrees=(2., 5.))
        a, b = self.state(batch=4, **settings), self.state(batch=4, **settings)
        torch.testing.assert_close(a.occupancy_grid(), b.occupancy_grid())
        sensor = a.lidar
        self.assertTrue(((sensor.noise_std >= .003) & (sensor.noise_std <= .012)).all())
        self.assertGreater(float(sensor.noise_std.max()-sensor.noise_std.min()), 0.)
        self.assertTrue(((sensor.dropout >= .03) & (sensor.dropout <= .1)).all())
        self.assertTrue(sensor.sector_mask.any(-1).all())
        self.assertTrue(torch.isinf(sensor.ranges[sensor.sector_mask]).all())
        other_noise = sensor.noise_std[1:].clone()
        other_sectors = sensor.sector_mask[1:].clone()
        grid = a.occupancy_grid()
        a.reset([0])
        torch.testing.assert_close(sensor.noise_std[1:], other_noise)
        torch.testing.assert_close(sensor.sector_mask[1:], other_sectors)
        torch.testing.assert_close(a.occupancy_grid()[1:], grid[1:])

    def test_outliers_are_bounded_and_do_not_create_motion(self):
        a = self.state(lidar_outlier_probability=1., lidar_outlier_error=.08)
        b = self.state()
        before = a.xy.clone()
        a.occupancy_grid(); b.occupancy_grid()
        delta = a.lidar.ranges - b.lidar.ranges
        self.assertLessEqual(float(delta.abs().max()), .080001)
        self.assertGreater(float(delta.abs().max()), .04)
        torch.testing.assert_close(a.xy, before)

    def test_clutter_sampler_has_thin_geometry_near_reachable_starts(self):
        cfg = NavigationCfg(**json.loads((Path(__file__).resolve().parents[1] /
                                          'scenarios/robot_lidar_clutter.json').read_text()))
        state = NavigationState(12, cfg, seed=42)
        state.reset()
        distances = ((state.obstacles[..., :2]-state.xy[:, None]).norm(dim=-1)
                     - state.obstacles[..., 2]).masked_fill(~state.valid, float('inf')).amin(-1)
        self.assertTrue(((distances >= .3) & (distances <= .7)).all())
        self.assertTrue((state.obstacles[..., 2][state.valid] < .07).any())
        self.assertTrue(((state.goal-state.xy).norm(dim=-1) >= 1.).all())
        self.assertTrue(((state.valid.sum(-1) >= 36) & (state.valid.sum(-1) <= 64)).all())
        from mentorpi_nav.teacher import plan_path
        for i in range(12):
            obstacles = state.obstacles[i][state.valid[i]].tolist()
            self.assertIsNotNone(plan_path(state.xy[i].tolist(), state.goal[i].tolist(), obstacles, cfg))


if __name__ == '__main__':
    unittest.main()
