"""Scenario coverage and complete-episode evaluation checks."""

from dataclasses import replace
import json
from pathlib import Path
import random
import unittest

import torch

from mentorpi_nav.core import NavigationCfg, NavigationState, sample_layout
from mentorpi_nav.evaluation import evaluate_actor
from mentorpi_nav.teacher import line_clear, plan_path, PathTeacher


class ScenarioTests(unittest.TestCase):
    def scenario(self, name):
        path = Path(__file__).resolve().parents[1] / "scenarios" / f"{name}.json"
        return replace(NavigationCfg(), **json.loads(path.read_text()))

    def test_varied_scenario_has_variable_counts_and_radii(self):
        state = NavigationState(32, self.scenario("varied"), seed=9)
        state.reset()
        counts = state.valid.sum(-1)
        self.assertGreater(int(counts.max()), int(counts.min()))
        self.assertTrue(((counts >= 2) & (counts <= 8)).all())
        radii = state.obstacles[..., 2][state.valid]
        self.assertTrue(((radii >= .12) & (radii <= .32)).all())
        self.assertGreater(float(radii.max() - radii.min()), .1)

    def test_passage_places_start_and_goal_across_divider(self):
        cfg = self.scenario("passages")
        rng = random.Random(19)
        for _ in range(12):
            start, goal, obstacles, yaw = sample_layout(cfg, rng)
            self.assertLess(start[0] * goal[0], 0)
            self.assertTrue(all(x == 0 for x, y, radius in obstacles))
            self.assertLessEqual(len(obstacles), cfg.max_obstacles)
            gap = min(abs(y) - radius for x, y, radius in obstacles) * 2
            self.assertGreaterEqual(gap, .55 - 1e-6)
            self.assertLessEqual(gap, .75 + 1e-6)

    def test_long_routes_enforce_distance(self):
        state = NavigationState(20, self.scenario("long_routes"), seed=11)
        state.reset()
        self.assertTrue((torch.linalg.vector_norm(state.goal - state.xy, dim=-1) >= 2.8).all())

    def test_clearance_reward_penalizes_wall_proximity(self):
        state = NavigationState(2, replace(self.scenario("safe_four"), obstacle_count=0))
        state.xy[:] = torch.tensor([[1.78, 0], [0, 0]])
        state.goal[:] = state.xy + torch.tensor([0., 1.])
        state.step(torch.zeros(2, 3))
        self.assertLess(float(state.reward[0]), float(state.reward[1]))

    def test_fixed_episode_evaluator_counts_timeouts_exactly(self):
        cfg = replace(NavigationCfg(), obstacle_count=0)
        def idle(observations):
            return torch.zeros(observations.batch_size[0], 3)
        result = evaluate_actor(idle, cfg, seed=33, episodes=7, episode_seconds=.1)
        self.assertEqual(result["timeouts"], 7)
        self.assertEqual(result["successes"] + result["collisions"] + result["timeouts"], 7)
        self.assertTrue(all(row["steps"] == 2 for row in result["episode_results"]))

    def test_teacher_routes_around_obstacle_with_continuous_clearance(self):
        cfg = NavigationCfg(obstacle_count=1)
        obstacles = [(0.,0.,.25)]
        path = plan_path((-1.,0.), (1.,0.), obstacles, cfg)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)
        for a,b in zip(path,path[1:]):
            self.assertTrue(line_clear(a,b,obstacles,cfg.robot_radius,cfg.arena_half_extent))

    def test_teacher_control_uses_body_frame(self):
        state = NavigationState(1, NavigationCfg(obstacle_count=0))
        state.goal[:] = torch.tensor([1.,0.])
        state.yaw[:] = torch.pi/2
        action = PathTeacher(state).actions(state)[0]
        self.assertAlmostEqual(float(action[0]),0.,places=5)
        self.assertLess(float(action[1]),0.)
        self.assertEqual(float(action[2]),0.)

    def test_default_sampler_still_matches_frozen_baseline_cases(self):
        path = Path(__file__).resolve().parents[1] / "logs/goal_baseline/seed_456.json"
        if not path.exists():
            self.skipTest("Local frozen baseline not available")
        report = json.loads(path.read_text())
        state = NavigationState(8,NavigationCfg(**report["navigation"]),seed=456)
        state.reset()
        for i,row in enumerate(report["episode_results"][:8]):
            self.assertEqual(state.xy[i].tolist(),row["start"])
            self.assertEqual(state.goal[i].tolist(),row["goal"])
            self.assertEqual(state.obstacles[i].tolist(),row["obstacles"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
