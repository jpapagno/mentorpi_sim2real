"""Checkpoint/playback integration and user control semantics."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mentorpi_nav.core import NavigationCfg
from mentorpi_nav.playback import NavigationPlayback
from view_navigation import checkpoint_choices

CHECKPOINT = Path(__file__).resolve().parents[1] / "logs/grid_empty_room/final.pt"


class PlaybackTests(unittest.TestCase):
    def test_checkpoint_browser_orders_iterations_numerically(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for name in ["model_100.pt", "model_25.pt", "model_9.pt", "model_broken.pt", "final.pt"]:
                (Path(directory) / name).touch()
            self.assertEqual([p.name for p in checkpoint_choices(directory)],
                             ["model_9.pt", "model_25.pt", "model_100.pt", "final.pt"])

    def test_manual_mode_without_checkpoint(self):
        session = NavigationPlayback(cfg=NavigationCfg(obstacle_count=0))
        session.state.xy[:] = 0
        session.state.yaw[:] = 0
        session.state.goal[:] = torch.tensor([1.0, 1.0])
        session.advance([0.0, 1.0, 0.0])
        self.assertGreater(float(session.state.xy[0, 1]), 0)
        self.assertEqual(session.total_steps, 1)
        self.assertEqual(session.mode, "manual")

    def test_timeout_holds_episode_until_reset(self):
        session = NavigationPlayback(cfg=NavigationCfg(obstacle_count=0))
        session.max_steps = 2
        session.advance()
        session.advance()
        self.assertEqual(session.status, "Timeout")
        session.advance([1.0, 0.0, 0.0])
        self.assertEqual(session.total_steps, 2)
        session.reset()
        self.assertFalse(session.terminal)
        self.assertEqual(session.episode_steps, 0)
        self.assertEqual(len(session.trail), 1)

    def test_goal_edit_rejects_wall_and_obstacle(self):
        session = NavigationPlayback()
        self.assertFalse(session.move_goal([2.1, 0.0]))
        self.assertFalse(session.move_goal(session.state.obstacles[0, 0, :2].tolist()))
        self.assertTrue(session.move_goal(session.state.xy[0].tolist()))

    def test_old_checkpoint_contract_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            checkpoint = Path(directory) / "final.pt"
            checkpoint.with_name("task.json").write_text(json.dumps({"model": "old_vector_model"}))
            with self.assertRaisesRegex(ValueError, "v2 occupancy-grid"):
                NavigationPlayback(checkpoint)

    @unittest.skipUnless(CHECKPOINT.exists(), "Requires locally trained checkpoint")
    def test_explicit_obstacle_scenario_updates_map_and_survives_resets(self):
        session = NavigationPlayback(CHECKPOINT, obstacle_count=4)
        self.assertEqual(session.training_cfg.obstacle_count, 0)
        self.assertEqual(session.state.cfg.obstacle_count, 4)
        self.assertEqual(session.state.valid.sum(), 4)
        with_obstacles = session.state.occupancy_grid().clone()
        session.state.valid[:] = False
        without_obstacles = session.state.occupancy_grid()
        self.assertGreater(float(with_obstacles.sum()), float(without_obstacles.sum()))
        session.reset()
        self.assertEqual(session.state.valid.sum(), 4)
        self.assertIn("checkpoint trained with: 0", session.scenario_description)
        session.advance()
        self.assertTrue(torch.isfinite(session.state.actions).all())

    def test_obstacle_override_uses_task_capacity_validation(self):
        with self.assertRaises(ValueError):
            NavigationPlayback(obstacle_count=9)
        with self.assertRaises(ValueError):
            NavigationPlayback(obstacle_count=-1)

    @unittest.skipUnless(CHECKPOINT.exists(), "Requires locally trained checkpoint")
    def test_trained_checkpoint_replays_finite_actions_with_original_settings(self):
        session = NavigationPlayback(CHECKPOINT)
        manifest = json.loads(CHECKPOINT.with_name("task.json").read_text())
        self.assertEqual(session.state.cfg.obstacle_count, manifest["navigation"]["obstacle_count"])
        session.state.xy[:] = 0
        for _ in range(5):
            session.advance()
            self.assertTrue(torch.isfinite(session.state.actions).all())
            self.assertTrue((session.state.actions.abs() <= 1).all())
            self.assertEqual(session.state.observations()["occupancy"].shape, (1, 1, 100, 100))
        self.assertEqual(session.mode, "policy")

    @unittest.skipUnless(CHECKPOINT.exists(), "Requires locally trained checkpoint")
    def test_checkpoint_cannot_silently_change_task_settings(self):
        with self.assertRaisesRegex(ValueError, "settings must match"):
            NavigationPlayback(CHECKPOINT, cfg=replace(NavigationCfg(), obstacle_count=8))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
