"""Goal-conditioned residual CNN for the unchanged grid/state input contract."""

import math

import torch
from torch import nn
from rsl_rl.models import CNNModel


class GoalResidualCNN(CNNModel):
    """Explicit goal branch + compact map branch, with a go-to-goal prior.

    The prior uses only the actor's supplied relative goal. Obstacle avoidance
    corrections are learned. Geometry searches/planning never run in this model.
    Constants match the task's 4m arena and 0.2m/s x/y limits.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs,obs_groups,obs_set,output_dim,**kwargs)
        self.action_model = output_dim == 3
        self.goal_encoder = nn.Sequential(nn.Linear(3,64),nn.ELU(),nn.Linear(64,64),nn.ELU())
        self.motion_encoder = nn.Sequential(nn.Linear(6,32),nn.ELU())
        self.map_encoder = nn.Sequential(nn.Linear(self.cnn_latent_dim,128),nn.ELU())
        if self.action_model:
            # Initial policy already moves toward its goal; demonstrations then
            # teach the correction rather than learning locomotion from scratch.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def _get_latent_dim(self):
        return 64+32+128

    def get_latent(self,obs,masks=None,hidden_state=None):
        raw=super().get_latent(obs,masks,hidden_state)
        goal=raw[:,:2]
        distance=torch.linalg.vector_norm(goal,dim=-1,keepdim=True)
        goal_features=torch.cat((goal/distance.clamp_min(1e-6),distance),dim=-1)
        return torch.cat((self.goal_encoder(goal_features),self.motion_encoder(raw[:,2:8]),
                          self.map_encoder(raw[:,8:])),dim=-1)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        value=self.mlp(self.get_latent(obs,masks,hidden_state))
        if self.action_model:
            goal=obs["policy"][:,:2]
            norm=torch.linalg.vector_norm(goal,dim=-1,keepdim=True)
            distance=norm*(4*math.sqrt(2))
            speed=(.9*distance/.2).clamp(max=.9)
            prior=torch.cat((goal/norm.clamp_min(1e-6)*speed,torch.zeros_like(norm)),dim=-1)
            value=value+prior
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(value)
                return self.distribution.sample()
            return self.distribution.deterministic_output(value)
        return value
