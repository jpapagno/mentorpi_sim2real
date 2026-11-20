"""TD3 with twin critics, delayed actor updates, target smoothing and replay.

Algorithm follows Fujimoto et al., https://arxiv.org/abs/1802.09477.
This implementation is separate from the existing RSL-RL occupancy policies.
"""
import copy
import numpy as np
import torch
from torch import nn
from .robot.laser_sectors import OBS_SIZE


class SectorActor(nn.Module):
    def __init__(self,translation_only=False):
        super().__init__()
        self.translation_only=translation_only
        self.net=nn.Sequential(nn.Linear(OBS_SIZE,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3),nn.Tanh())

    def forward(self,observation):
        action=self.net(observation)
        if self.translation_only:
            action=torch.cat((action[:,:2],torch.zeros_like(action[:,2:])),dim=-1)
        return action


class SectorPolicy(nn.Module):
    """Mapping adapter shared by existing playback and evaluation."""
    def __init__(self,actor):
        super().__init__();self.actor=actor

    def forward(self,observations):return self.actor(observations['policy'])


class TwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        def net():return nn.Sequential(nn.Linear(OBS_SIZE+3,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q1,self.q2=net(),net()

    def forward(self,state,action):
        features=torch.cat((state,action),-1)
        return self.q1(features),self.q2(features)


class ReplayBuffer:
    def __init__(self,capacity,seed=42):
        self.capacity=capacity;self.position=0;self.size=0;self.rng=np.random.default_rng(seed)
        self.state=np.empty((capacity,OBS_SIZE),np.float32)
        self.next_state=np.empty_like(self.state)
        self.action=np.empty((capacity,3),np.float32)
        self.reward=np.empty((capacity,1),np.float32)
        self.terminal=np.empty((capacity,1),np.float32)

    def add(self,state,action,reward,next_state,terminal):
        values=[v.detach().cpu().numpy() for v in (state,action,reward.reshape(-1,1),next_state,terminal.reshape(-1,1))]
        n=len(state)
        if n>self.capacity:raise ValueError('Replay batch exceeds capacity')
        index=(self.position+np.arange(n))%self.capacity
        for target,value in zip((self.state,self.action,self.reward,self.next_state,self.terminal),values):target[index]=value
        self.position=(self.position+n)%self.capacity;self.size=min(self.size+n,self.capacity)

    def sample(self,batch_size,device):
        if self.size<batch_size:raise ValueError('Insufficient replay samples')
        index=self.rng.integers(self.size,size=batch_size)
        return [torch.as_tensor(v[index],device=device) for v in (self.state,self.action,self.reward,self.next_state,self.terminal)]


class TD3:
    def __init__(self,device='cpu',gamma=.995,lr=.0003,tau=.005,policy_noise=.2,noise_clip=.5,policy_delay=2,translation_only=False):
        self.device=device;self.gamma=gamma;self.tau=tau
        self.policy_noise=policy_noise;self.noise_clip=noise_clip;self.policy_delay=policy_delay;self.updates=0
        self.actor=SectorActor(translation_only).to(device);self.critic=TwinCritic().to(device)
        self.actor_target=copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target=copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

    def update(self,batch):
        state,action,reward,next_state,terminal=batch
        with torch.no_grad():
            noise=(torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip,self.noise_clip)
            next_action=(self.actor_target(next_state)+noise).clamp(-1,1)
            if self.actor.translation_only:next_action[:,2]=0
            q1,q2=self.critic_target(next_state,next_action)
            target=reward+(1-terminal)*self.gamma*torch.minimum(q1,q2)
        q1,q2=self.critic(state,action)
        loss=(q1-target).square().mean()+(q2-target).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite TD3 critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True);loss.backward();self.critic_optimizer.step()
        self.updates+=1
        metrics={'critic_loss':float(loss.detach()),'q_mean':float(q1.detach().mean())}
        if self.updates%self.policy_delay==0:
            self.critic.requires_grad_(False)
            actor_loss=-self.critic(state,self.actor(state))[0].mean()
            self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward();self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            with torch.no_grad():
                for target_net,source in ((self.actor_target,self.actor),(self.critic_target,self.critic)):
                    for target_param,param in zip(target_net.parameters(),source.parameters()):target_param.lerp_(param,self.tau)
            metrics['actor_loss']=float(actor_loss.detach())
        return metrics

    def save(self,path,transitions):
        from pathlib import Path
        path=Path(path);temporary=path.with_suffix('.tmp')
        torch.save({'actor_state_dict':self.actor.state_dict(),'critic_state_dict':self.critic.state_dict(),
                    'actor_target':self.actor_target.state_dict(),'critic_target':self.critic_target.state_dict(),
                    'actor_optimizer':self.actor_optimizer.state_dict(),'critic_optimizer':self.critic_optimizer.state_dict(),
                    'updates':self.updates,'transitions':transitions,'translation_only':self.actor.translation_only},temporary)
        temporary.replace(path)

    def restore(self,saved):
        if saved.get('translation_only',False)!=self.actor.translation_only:raise ValueError('Checkpoint action contract differs')
        self.actor.load_state_dict(saved['actor_state_dict'],strict=True)
        self.critic.load_state_dict(saved['critic_state_dict'],strict=True)
        self.actor_target.load_state_dict(saved['actor_target'],strict=True)
        self.critic_target.load_state_dict(saved['critic_target'],strict=True)
        self.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        self.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        self.updates=int(saved['updates'])
