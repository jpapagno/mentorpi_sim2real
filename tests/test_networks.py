"""Residual policy contract, including goal steering and checkpoint playback."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tensordict import TensorDict

from mentorpi_nav.core import NavigationCfg, NavigationState
from mentorpi_nav.model_spec import CNN_CONFIG, HIDDEN_DIMS
from mentorpi_nav.networks import GoalResidualCNN
from mentorpi_nav.playback import load_actor


class ResidualPolicyTests(unittest.TestCase):
    def test_prior_and_roundtrip(self):
        state = NavigationState(4, NavigationCfg(obstacle_count=0))
        state.xy.zero_()
        state.yaw.zero_()
        state.goal[:] = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        obs = TensorDict(state.observations(), batch_size=[4])
        actor = GoalResidualCNN(obs, {"actor":["policy","occupancy"]}, "actor", 3,
            hidden_dims=HIDDEN_DIMS, activation="elu", obs_normalization=False,
            cnn_cfg=CNN_CONFIG, distribution_cfg={"class_name":"GaussianDistribution","init_std":.15})
        with torch.no_grad():
            actions = actor(obs)
        torch.testing.assert_close(actions, torch.tensor([[.9,0.,0.],[-.9,0.,0.],[0.,.9,0.],[0.,-.9,0.]]))
        actor(obs, stochastic_output=True)
        self.assertTrue(torch.isfinite(actor.distribution.entropy).all())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            checkpoint = Path(directory)/"final.pt"
            checkpoint.with_name("task.json").write_text(json.dumps({"network_architecture":"goal_residual_cnn_v1"}))
            torch.save({"actor_state_dict":actor.state_dict()}, checkpoint)
            restored = load_actor(checkpoint, dict(obs))
            with torch.no_grad():
                torch.testing.assert_close(restored(obs), actions)


if __name__ == "__main__":
    unittest.main()
