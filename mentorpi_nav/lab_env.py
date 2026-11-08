"""Isaac Lab 3 direct environment with analytic navigation and USD visuals.

The analytic model owns motion/collision. USD objects are visuals only; no wheel
articulation, sensor rendering, or hidden dependence on PhysX contact events.
"""

from dataclasses import asdict
from pathlib import Path

import torch
import gymnasium as gym
import numpy as np
from pxr import Gf, UsdGeom, UsdLux

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

from .core import NavigationCfg, NavigationState


@configclass
class MentorPiEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 1
    episode_length_s = 40.0
    action_space = 3
    observation_space = 8
    state_space = 0
    ui_window_class_type = None
    navigation = NavigationCfg()
    asset_path: str | None = None
    visual_envs: int = 1
    sim = SimulationCfg(dt=0.05, render_interval=1,
                        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False))
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=5.0)


class MentorPiEnv(DirectRLEnv):
    cfg: MentorPiEnvCfg

    def __init__(self, cfg, **kwargs):
        cfg.observation_space = cfg.navigation.observation_size
        cfg.sim.dt = cfg.navigation.dt
        cfg.episode_length_s = cfg.navigation.episode_seconds
        if cfg.decimation != 1:
            raise ValueError("Analytic task requires decimation=1")
        if cfg.asset_path and not Path(cfg.asset_path).is_file():
            raise FileNotFoundError(cfg.asset_path)
        super().__init__(cfg, **kwargs)
        self.nav = NavigationState(self.num_envs, cfg.navigation, self.device, cfg.seed)
        self.nav.reset()

    def _setup_scene(self):
        stage = self.sim.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        light = UsdLux.DistantLight.Define(stage, "/World/Sun")
        light.CreateIntensityAttr(2500)
        self._visuals = []
        # Only draw a few environments; every training environment still has
        # its own independent tensor state and collision geometry.
        for i in range(min(self.cfg.visual_envs, self.num_envs)):
            root = f"/World/Navigation_{i}"
            offset = Gf.Vec3d(i * self.cfg.scene.env_spacing, 0, 0)
            UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(offset)
            h = self.cfg.navigation.arena_half_extent
            self._shape(stage, root + "/Floor", "cube", (0, 0, -0.025), (2*h, 2*h, 0.05), (0.22, 0.25, 0.28))
            for j, (pos, size) in enumerate([
                ((h, 0, 0.15), (0.03, 2*h, 0.3)), ((-h, 0, 0.15), (0.03, 2*h, 0.3)),
                ((0, h, 0.15), (2*h, 0.03, 0.3)), ((0, -h, 0.15), (2*h, 0.03, 0.3))]):
                self._shape(stage, f"{root}/Wall{j}", "cube", pos, size, (0.45, 0.45, 0.5))
            robot = UsdGeom.Xform.Define(stage, root + "/Robot")
            robot_pos = robot.AddTranslateOp()
            robot_rot = robot.AddRotateZOp()
            if self.cfg.asset_path:
                visual = stage.DefinePrim(root + "/Robot/Visual", "Xform")
                visual.GetReferences().AddReference(str(Path(self.cfg.asset_path).resolve()).replace("\\", "/"))
                variants = visual.GetVariantSets().GetVariantSet("Physics")
                if "none" in variants.GetVariantNames():
                    variants.SetVariantSelection("none")
                # This imported model's lowest point is ~19 mm above its origin.
                # Keep source transforms intact; floor alignment is adjusted below.
                bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeLocalBound(visual).ComputeAlignedRange()
                visual_xf = UsdGeom.Xformable(visual)
                visual_xf.AddTranslateOp(opSuffix="floorAlign").Set(Gf.Vec3d(0, 0, -bounds.GetMin()[2]))
                for prim in stage.Traverse():
                    if prim.GetPath().HasPrefix(visual.GetPath()) and any(
                        "RigidBodyAPI" in schema or "ArticulationRootAPI" in schema
                        or "CollisionAPI" in schema for schema in prim.GetAppliedSchemas()
                    ):
                        raise ValueError("Visual asset still has physics enabled; use its Physics=none variant")
            else:
                self._shape(stage, root + "/Robot/Body", "cylinder", (0, 0, 0.07),
                            (self.cfg.navigation.robot_radius, 0.14), (0.1, 0.5, 0.9))
            self._shape(stage, root + "/Robot/Heading", "cube", (0.13, 0, 0.17),
                        (0.1, 0.02, 0.02), (0.2, 0.9, 1.0))
            goal = self._shape(stage, root + "/Goal", "sphere", (0, 0, 0.04),
                               (self.cfg.navigation.goal_tolerance,), (0.15, 0.9, 0.25))
            obstacle_ops = []
            for j in range(self.cfg.navigation.max_obstacles):
                obstacle_ops.append(self._shape(stage, f"{root}/Obstacle{j}", "cylinder", (0, 0, -2),
                    (self.cfg.navigation.obstacle_radius, 0.4), (0.85, 0.25, 0.12)))
            self._visuals.append((robot_pos, robot_rot, goal, obstacle_ops))

    @staticmethod
    def _shape(stage, path, kind, position, size, color):
        if kind == "cube":
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*size))
        elif kind == "cylinder":
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateRadiusAttr(size[0])
            shape.CreateHeightAttr(size[1])
        else:
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(size[0])
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.Xformable(shape)
        translate = xform.AddTranslateOp()
        translate.Set(Gf.Vec3d(*position))
        ops = xform.GetOrderedXformOps()
        xform.SetXformOpOrder([translate] + [op for op in ops if op != translate])
        return shape.GetPrim()

    def _sync_visuals(self):
        # No GPU-to-CPU copies or USD edits during headless training.
        if not self.sim.visualizers:
            return
        xy = self.nav.xy[:len(self._visuals)].cpu().tolist()
        yaw = self.nav.yaw[:len(self._visuals)].cpu().tolist()
        goals = self.nav.goal[:len(self._visuals)].cpu().tolist()
        obstacles = self.nav.obstacles[:len(self._visuals)].cpu().tolist()
        for i, (pos_op, rot_op, goal, obs) in enumerate(self._visuals):
            pos_op.Set(Gf.Vec3d(*xy[i], 0))
            rot_op.Set(yaw[i] * 180 / torch.pi)
            self._move_prim(goal, (*goals[i], 0.04))
            for j, prim in enumerate(obs):
                visible = bool(self.nav.valid[i, j])
                self._move_prim(prim, (*obstacles[i][j][:2], 0.2) if visible else (0, 0, -2))
                UsdGeom.Cylinder(prim).GetRadiusAttr().Set(obstacles[i][j][2] if visible else self.cfg.navigation.obstacle_radius)

    @staticmethod
    def _move_prim(prim, position):
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*position))

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()

    def _apply_action(self):
        self.nav.step(self.actions)
        self._sync_visuals()

    def _get_observations(self):
        return self.nav.observations()

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        n = self.cfg.navigation.occupancy_size
        self.single_observation_space["occupancy"] = gym.spaces.Box(0, 1, (1, n, n), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

    def _get_rewards(self):
        return self.nav.reward.clone()

    def _get_dones(self):
        self.outcomes = {"collision": self.nav.collision.clone(), "success": self.nav.success.clone()}
        terminated = self.nav.collision | self.nav.success
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        return terminated, timeout

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "nav") and len(env_ids):
            self.extras["log"] = {
                "Metrics/success": self.nav.success[env_ids].float().mean(),
                "Metrics/collision": self.nav.collision[env_ids].float().mean(),
            }
        super()._reset_idx(env_ids)
        if hasattr(self, "nav"):
            self.nav.reset(env_ids)
            self._sync_visuals()

    def task_manifest(self):
        return {"navigation": asdict(self.cfg.navigation), "observation_size": self.cfg.observation_space,
                "occupancy_shape": [1, self.cfg.navigation.occupancy_size, self.cfg.navigation.occupancy_size],
                "occupancy_frame": "robot_forward_up_left_left", "occupancy_values": {"free": 0, "occupied": 1},
                "asset_path": self.cfg.asset_path, "model": "analytic_swept_disc_occupancy_v2"}
