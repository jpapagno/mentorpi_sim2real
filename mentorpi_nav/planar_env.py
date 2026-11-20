"""Isaac-free navigation adapters: Gymnasium for tools, TensorDict for PPO."""
from dataclasses import asdict
import math

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from .core import NavigationCfg, NavigationState


def task_manifest(cfg):
    return {'navigation':asdict(cfg),'observation_size':8,'occupancy_shape':[1,100,100],
            'occupancy_frame':'robot_forward_up_left_left','occupancy_values':{'free':0,'occupied':1},
            'occupancy_source':cfg.occupancy_source,'model':'analytic_swept_disc_occupancy_v2',
            'backend':'planar_torch','asset_path':None,
            'collision_footprint':'oriented_rectangle' if cfg.robot_size is not None else 'disc'}


class PlanarBatch:
    """Batched tensor task; termination is captured before optional autoreset."""
    def __init__(self, cfg=None, num_envs=1, device='cpu', seed=42):
        self.cfg = cfg or NavigationCfg()
        self.nav = NavigationState(num_envs,self.cfg,device,seed)
        self.device = self.nav.device
        self.num_envs = num_envs
        self.max_episode_length = math.ceil(self.cfg.episode_seconds/self.cfg.dt)
        self.episode_length_buf = torch.zeros(num_envs,device=device,dtype=torch.long)
        self.reset()

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs,device=self.device)
        self.nav.reset(ids)
        self.episode_length_buf[ids] = 0
        return self.nav.observations()

    def step(self, actions, autoreset=False):
        self.nav.step(actions)
        self.episode_length_buf += 1
        reward = self.nav.reward.clone()
        success, collision = self.nav.success.clone(), self.nav.collision.clone()
        terminated = success | collision
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        info = {'success':success,'collision':collision,'time_outs':truncated.clone()}
        done = terminated | truncated
        if autoreset and done.any():
            info['final_indices'] = done.nonzero().flatten()
            # Like the Isaac adapter, reset before acquiring the next scan.
            # Sampling a discarded terminal scan would advance sensor RNG and
            # change the next episode's observations.
            self.reset(info['final_indices'])
        return self.nav.observations(),reward,terminated,truncated,info


class PlanarGymEnv(gym.Env):
    """Standard single-environment NumPy Gymnasium API (explicit reset)."""
    metadata = {'render_modes':[]}

    def __init__(self, cfg=None, seed=42):
        self.cfg = cfg or NavigationCfg()
        self.batch = PlanarBatch(self.cfg,seed=seed)
        self.action_space = gym.spaces.Box(-1.,1.,(3,),dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'policy':gym.spaces.Box(-np.inf,np.inf,(8,),dtype=np.float32),
            'occupancy':gym.spaces.Box(0.,1.,(1,100,100),dtype=np.float32)})
        self._done = False

    @staticmethod
    def numpy_observation(obs):
        return {k:v[0].cpu().numpy().copy() for k,v in obs.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.batch = PlanarBatch(self.cfg,seed=seed)
            obs = self.batch.nav.observations()
        else:
            obs = self.batch.reset()
        self._done = False
        return self.numpy_observation(obs),{}

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset after termination before stepping again')
        obs,reward,terminated,truncated,info = self.batch.step(torch.as_tensor(action,dtype=torch.float32).reshape(1,3))
        self._done = bool(terminated[0] | truncated[0])
        return self.numpy_observation(obs),float(reward[0]),bool(terminated[0]),bool(truncated[0]),{k:bool(v[0]) for k,v in info.items()}


class RslPlanarEnv:
    """RSL-RL VecEnv protocol without importing Isaac or copying grids to CPU."""
    def __init__(self, cfg=None, num_envs=16, device='cpu', seed=42):
        self.batch = PlanarBatch(cfg,num_envs,device,seed)
        self.cfg = asdict(self.batch.cfg)
        self.num_envs,self.num_actions,self.device = num_envs,3,self.batch.device
        self.max_episode_length = self.batch.max_episode_length

    @property
    def episode_length_buf(self):
        return self.batch.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self,value):
        self.batch.episode_length_buf = value

    def get_observations(self):
        return TensorDict(self.batch.nav.observations(),batch_size=[self.num_envs])

    def step(self,actions):
        obs,reward,terminated,truncated,info = self.batch.step(actions.clamp(-1,1),autoreset=True)
        done = terminated | truncated
        extras = {'time_outs':truncated,'outcomes':{'success':info['success'],'collision':info['collision']}}
        if done.any():
            extras['log']={'Metrics/success':info['success'][done].float().mean(),
                           'Metrics/collision':info['collision'][done].float().mean()}
        return TensorDict(obs,batch_size=[self.num_envs]),reward,done.long(),extras

    def close(self):
        pass
