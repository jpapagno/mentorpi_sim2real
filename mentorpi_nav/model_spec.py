"""Shared network layout for training and lightweight policy playback."""

HIDDEN_DIMS = [128, 128]
CNN_CONFIG = {
    "output_channels": [8, 16, 16],
    "kernel_size": [8, 4, 3],
    "stride": [4, 2, 1],
    "activation": "elu",
}


def planar_ppo_config(seed=42, device='cpu', rollout_steps=32, save_every=25, learning_rate=1e-4,
                      residual=True, gamma=.99, entropy_coef=.001):
    """RSL-RL configuration with no Isaac configuration-class dependency."""
    import copy
    import math
    if not 0 < gamma < 1:
        raise ValueError('gamma must be between zero and one')
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError('entropy coefficient must be finite and nonnegative')
    model = {'class_name':'mentorpi_nav.networks:GoalResidualCNN' if residual else 'CNNModel',
             'hidden_dims':HIDDEN_DIMS,'activation':'elu','obs_normalization':False,
             'cnn_cfg':CNN_CONFIG}
    actor = copy.deepcopy(model)
    actor['distribution_cfg']={'class_name':'GaussianDistribution','init_std':.5}
    return {'seed':seed,'device':device,'num_steps_per_env':rollout_steps,'save_interval':save_every,
            'obs_groups':{'actor':['policy','occupancy'],'critic':['policy','occupancy']},
            'clip_actions':1.,'check_for_nan':True,'logger':'tensorboard',
            'actor':actor,'critic':copy.deepcopy(model),
            'algorithm':{'class_name':'PPO','value_loss_coef':1.,'use_clipped_value_loss':True,
                         'clip_param':.2,'entropy_coef':entropy_coef,'num_learning_epochs':4,'num_mini_batches':4,
                         'learning_rate':learning_rate,'schedule':'fixed','gamma':gamma,'lam':.95,
                         'desired_kl':.01,'max_grad_norm':1.}}
