"""Experimental stateful yaw response for replay, not a motor-board diagnosis.

A bounded signed drive-memory state models slower breakaway after reversal,
faster same-direction restarts, finite acceleration and fast zero-command stop.
This is a phenomenological model. It is not enabled in navigation/training by
default and does not send commands or alter the deployed robot.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class YawResponseCfg:
    breakaway_memory: float = .08
    rise_time_s: float = .25
    stop_time_s: float = .04
    rate_gain: float = 1.
    idle_memory_decay_s: float = 10.
    minimum_command_radps: float = .205

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (self.breakaway_memory,self.rise_time_s,
                self.stop_time_s,self.rate_gain,self.idle_memory_decay_s)):
            raise ValueError('Response parameters must be finite and positive')
        if not 0<=self.minimum_command_radps<.3:raise ValueError('Invalid minimum yaw command')


class YawResponse:
    def __init__(self,cfg=None):
        self.cfg=cfg or YawResponseCfg();self.memory=0.;self.rate=0.;self.angle=0.

    def step(self,command,dt):
        if not math.isfinite(command) or abs(command)>.300001 or not math.isfinite(dt) or not 0<dt<=.05:
            raise ValueError('Replay requires bounded yaw command and timestep')
        cfg=self.cfg;command=max(-.3,min(.3,command))
        if abs(command)>1e-6:
            self.memory=max(-cfg.breakaway_memory,min(cfg.breakaway_memory,self.memory+command*dt))
        else:self.memory*=math.exp(-dt/cfg.idle_memory_decay_s)
        engaged=abs(command)>=cfg.minimum_command_radps and self.memory*command>0 and abs(self.memory)>=cfg.breakaway_memory-1e-7
        target=cfg.rate_gain*command if engaged else 0.
        tau=cfg.rise_time_s if engaged else cfg.stop_time_s
        old=self.rate;self.rate=target+(self.rate-target)*math.exp(-dt/tau)
        self.angle+=(old+self.rate)*.5*dt
        return self.rate


class BatchedYawResponse:
    """Torch equivalent used by the shared simulator; sensor/control frames unchanged."""
    def __init__(self,num_envs,device,cfg=None):
        import torch
        self.cfg=cfg or YawResponseCfg()
        self.memory=torch.zeros(num_envs,device=device)
        self.rate=torch.zeros_like(self.memory)

    def reset(self,ids):self.memory[ids]=0.;self.rate[ids]=0.

    def step(self,command,dt):
        import torch
        cfg=self.cfg
        self.memory=torch.where(command.abs()>1e-6,(self.memory+command*dt).clamp(-cfg.breakaway_memory,cfg.breakaway_memory),
                                self.memory*math.exp(-dt/cfg.idle_memory_decay_s))
        engaged=(command.abs()>=cfg.minimum_command_radps)&(self.memory*command>0)&(self.memory.abs()>=cfg.breakaway_memory-1e-7)
        target=torch.where(engaged,cfg.rate_gain*command,0.)
        decay=torch.where(engaged,math.exp(-dt/cfg.rise_time_s),math.exp(-dt/cfg.stop_time_s))
        self.rate=target+(self.rate-target)*decay
        return self.rate
