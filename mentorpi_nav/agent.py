"""Small RSL-RL PPO agent for normalized geometry observations."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .model_spec import CNN_CONFIG, HIDDEN_DIMS


@configclass
class MentorPiPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 500
    save_interval = 25
    experiment_name = "mentorpi_navigation"
    obs_groups = {"actor": ["policy", "occupancy"], "critic": ["policy", "occupancy"]}
    clip_actions = 1.0
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(**CNN_CONFIG),
        hidden_dims=HIDDEN_DIMS, activation="elu", obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.5))
    critic = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(**CNN_CONFIG),
        hidden_dims=HIDDEN_DIMS, activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=4, num_mini_batches=4,
        learning_rate=3e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0)
