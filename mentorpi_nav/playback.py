"""Policy playback using the training task's exact analytic model, without Isaac.

Only PyTorch, TensorDict, and RSL-RL's model code are needed. No simulator,
USD, Newton, renderer, optimizer, or critic is initialized.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import CNNModel

from .core import NavigationCfg, NavigationState, swept_collision
from .model_spec import CNN_CONFIG, HIDDEN_DIMS

MODEL_ID = "analytic_swept_disc_occupancy_v2"


def load_actor(checkpoint, observations, device="cpu"):
    manifest=json.loads(Path(checkpoint).with_name("task.json").read_text(encoding="utf-8"))
    if manifest.get('network_architecture')=='laser_sector_td3_v1':
        from .td3 import SectorActor,SectorPolicy
        from .robot.laser_sectors import CONTRACT,OBS_SIZE
        if manifest.get('observation_contract')!=CONTRACT or observations['policy'].shape[-1]!=OBS_SIZE:
            raise ValueError('Incompatible sector observation contract')
        translation_only=manifest.get('navigation',{}).get('translation_only',False)
        actor=SectorActor(translation_only).to(device)
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        if saved.get('translation_only',False)!=translation_only:
            raise ValueError('Checkpoint/task translation-only mismatch')
        actor.load_state_dict(saved['actor_state_dict'],strict=True)
        return SectorPolicy(actor).eval()
    model_class=CNNModel
    if manifest.get("network_architecture") == "goal_residual_cnn_v1":
        navigation = manifest.get("navigation", {})
        if (navigation.get('goal_normalization_half_extent') or navigation.get("arena_half_extent",2.0)) != 2.0 or tuple(navigation.get("speed_limits", (.2,.2,.5))) != (.2,.2,.5):
            raise ValueError("Residual v1 prior requires the 4 m arena and original speed limits")
        from .networks import GoalResidualCNN
        model_class=GoalResidualCNN
    actor = model_class(
        TensorDict(observations, batch_size=[observations["policy"].shape[0]]),
        {"actor": ["policy", "occupancy"]}, "actor", 3,
        hidden_dims=HIDDEN_DIMS, activation="elu", obs_normalization=False,
        cnn_cfg=CNN_CONFIG,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    ).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.eval()
    return actor


class NavigationPlayback:
    """One-environment session that stops on arrival, collision, or timeout."""

    def __init__(self, checkpoint=None, cfg=None, seed=123, device="cpu", obstacle_count=None, scenario_overrides=None):
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.obstacle_override = obstacle_count
        self.scenario_overrides = scenario_overrides
        if self.checkpoint:
            manifest = json.loads(self.checkpoint.with_name("task.json").read_text(encoding="utf-8"))
            sectors=manifest.get('network_architecture')=='laser_sector_td3_v1'
            if manifest["model"] != MODEL_ID or (not sectors and manifest.get("occupancy_shape") != [1, 100, 100]):
                raise ValueError("This viewer requires a v2 occupancy-grid checkpoint and its task.json")
            loaded_cfg = NavigationCfg(**manifest["navigation"])
            if cfg is not None and cfg != loaded_cfg:
                raise ValueError("Viewer settings must match the checkpoint task")
            cfg = loaded_cfg
        self.training_cfg = cfg or NavigationCfg()
        cfg = replace(self.training_cfg,**scenario_overrides) if scenario_overrides is not None else self.training_cfg
        if scenario_overrides is not None and self.checkpoint:
            invariant=('policy_input','arena_half_extent','robot_radius','robot_size','goal_normalization_half_extent','speed_limits','acceleration_limits','dt',
                       'occupancy_size','occupancy_resolution','occupancy_inflation','goal_tolerance')
            for name in invariant:
                before,after=getattr(self.training_cfg,name),getattr(cfg,name)
                if isinstance(before,(list,tuple)):before,after=list(before),list(after)
                if before != after:raise ValueError('Challenge scenario must preserve policy contract: '+name)
        if obstacle_count is not None:
            cfg=replace(cfg,obstacle_count=obstacle_count)
        self.state = NavigationState(1, cfg, device=device, seed=seed)
        self.state.reset()
        self.actor = load_actor(self.checkpoint, self.state.observations(), device) if self.checkpoint else None
        self.mode = "policy" if self.actor is not None else "manual"
        self.max_steps = math.ceil(self.state.cfg.episode_seconds / self.state.cfg.dt)
        self.total_steps = 0
        self.reset_counters()

    def reset_counters(self):
        self.episode_steps = 0
        self.episode_return = 0.0
        self.status = "Running"
        self.trail = [tuple(self.state.xy[0].cpu().tolist())]

    def reset(self):
        self.state.reset()
        self.reset_counters()

    @property
    def terminal(self):
        return self.status in ("Goal reached", "Collision", "Timeout")

    @property
    def scenario_description(self):
        actual = int(self.state.valid[0].sum())
        polygons = int((self.state.polygon_counts[0]>0).sum())
        if self.state.cfg.scene_spec is not None:
            prefix='Challenge scenario | ' if self.checkpoint and self.scenario_overrides is not None else ''
            return prefix+f'Mixed-shape scene: {actual} circles, {polygons} polygons | layout: {self.state.cfg.scene_spec.get("layout","fixed")}'
        if self.checkpoint:
            trained = self.training_cfg.obstacle_count if self.training_cfg.min_obstacles is None else f"{self.training_cfg.min_obstacles}–{self.training_cfg.obstacle_count}"
            return f"Evaluation obstacles: {actual} | checkpoint trained with: {trained}"
        return f"Playground obstacles: {actual}"

    def advance(self, manual_action=None):
        if self.terminal:
            return
        # Playback state is also edited by GUI callbacks (reset / goal changes).
        # no_grad keeps it mutable outside this scope, unlike inference tensors.
        with torch.no_grad():
            if self.mode == "policy":
                if self.actor is None:
                    raise ValueError("Load a checkpoint to use policy mode")
                obs = TensorDict(self.state.observations(), batch_size=[1])
                action = self.actor(obs)
            elif self.mode == "random":
                action = torch.rand(1, 3, device=self.state.device) * 2 - 1
            else:
                action = torch.tensor([manual_action or [0.0, 0.0, 0.0]], device=self.state.device)
            self.state.step(action)
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_return += float(self.state.reward[0])
        self.trail.append(tuple(self.state.xy[0].cpu().tolist()))
        self.trail = self.trail[-1000:]
        if self.state.collision[0]:
            self.status = "Collision"
        elif self.state.success[0]:
            self.status = "Goal reached"
        elif self.episode_steps >= self.max_steps:
            self.status = "Timeout"

    def move_goal(self, xy):
        point = torch.tensor([xy], device=self.state.device)
        if self.state.cfg.scene_spec is not None:
            from .geometry import polygon_sweep
            if polygon_sweep(point,point,self.state.polygons,self.state.polygon_counts,self.state.cfg.robot_radius).item():
                return False
        if swept_collision(point, point, self.state.obstacles, self.state.valid,
                           self.state.cfg.robot_radius, self.state.cfg.arena_half_extent).item():
            return False
        self.state.goal[:] = point
        self.state.success[:] = False
        if self.terminal:
            self.state.collision[:] = False
            self.reset_counters()
        return True
