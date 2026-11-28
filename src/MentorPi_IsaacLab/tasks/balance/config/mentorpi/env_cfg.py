# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import cast

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg

from isaaclab_tasks.utils import PresetCfg

from .....assets.robots.mentorpi import MENTORPI_LIDAR_CFG, MENTORPI_RECT_CFG

from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCaster, MultiMeshRayCasterCfg, MultiMeshRayCasterData


@configclass
class MotionPhysicsCfg(PresetCfg):
    """Physics presets for the motion environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=64,
            nconmax=32,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


##
# Scene definition
##

GOAL_DISTANCE_RANGE = (0.9, 1.1)
OBSTACLE_DISTANCE_RANGE = (0.4, 0.5)
OBSTACLE_SIZE = (0.35 / 3, 0.35 / 3, 0.20 / 3)
GOAL_HALF_WIDTH = 0.15
ARENA_HALF_WIDTH = 1.75


def obstacle_block(prim_name: str, x: float, y: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_SIZE,
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
            collision_props=[sim_utils.UsdPhysicsCollisionCfg()],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.32, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, OBSTACLE_SIZE[2] / 2)),
    )


@configclass
class MotionSceneCfg(InteractiveSceneCfg):
    """Configuration for a motion scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: RigidObjectCfg = MENTORPI_RECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # obstacles
    obstacle_east = obstacle_block("ObstacleEast", 0.9, 0.0)
    obstacle_north = obstacle_block("ObstacleNorth", 0.0, 0.9)
    obstacle_west = obstacle_block("ObstacleWest", -0.9, 0.0)

    # Obstacle lidar and contact sensors
    lidar: MultiMeshRayCasterCfg = MENTORPI_LIDAR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # Keep the horizontal scan below the shortened obstacles' tops.
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.01)),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/ObstacleEast"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/ObstacleNorth"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/ObstacleWest"),
        ],
        debug_vis=True,
    )

    obstacle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/ObstacleEast",
            "{ENV_REGEX_NS}/ObstacleNorth",
            "{ENV_REGEX_NS}/ObstacleWest",
        ],
        history_length=2,  # Covers both physics steps at decimation=2
    )

    # goal
    goal = VisualizationMarkersCfg(
        prim_path="/Visuals/MentorPiGoals",
        markers={
            "goal": sim_utils.CuboidCfg(
                size=(2 * GOAL_HALF_WIDTH, 2 * GOAL_HALF_WIDTH, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.8, 0.25)),
            ),
        },
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


class PlanarVelocityAction(ActionTerm):
    """Command the block's world-frame X and Y velocity."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((env.num_envs, 2), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        if self.cfg.check_action_bounds:
            limit = self.cfg.speed_limit + 1e-6
            if not bool((actions.abs() <= limit).all()):
                raise ValueError(
                    f"Raw XY actions must be finite and within ±{self.cfg.speed_limit} m/s. "
                    f"Batch min={actions.amin(dim=0).tolist()}, max={actions.amax(dim=0).tolist()}."
                )
            if self._env.common_step_counter % 1000 == 0:
                print(f"Raw XY actions (m/s) min={actions.amin(dim=0).tolist()}, max={actions.amax(dim=0).tolist()}")
        self._raw_actions.copy_(actions)
        # The simulated robot holds its spawn heading, so policy and body XY are aligned.
        velocity = actions * self.cfg.speed_scale
        self._processed_actions.copy_(velocity.clamp(-self.cfg.speed_limit, self.cfg.speed_limit))

    def apply_actions(self):
        pose = self._asset.data.root_link_pose_w.torch.clone()
        velocity = self._asset.data.root_vel_w.torch.clone()
        # Restore the spawn orientation after any rotation from ground contact.
        pose[:, 3:] = self._asset.data.default_root_pose.torch[:, 3:]
        velocity[:, :2] = self._processed_actions
        velocity[:, 3:] = 0.0
        self._asset.write_root_pose_to_sim_index(root_pose=pose)
        self._asset.write_root_velocity_to_sim_index(root_velocity=velocity)

    def reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self._raw_actions[indices] = 0.0
        self._processed_actions[indices] = 0.0


@configclass
class PlanarVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PlanarVelocityAction
    speed_limit: float = 0.6
    speed_scale: float = 1.0
    check_action_bounds: bool = False


@configclass
class ActionsCfg:
    planar_velocity = PlanarVelocityActionCfg(
        asset_name="robot",
    )


def goal_positions_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the per-environment goal positions, including before the first reset."""
    positions = getattr(env, "_mentorpi_goal_positions_w", None)
    if positions is None:
        origins = env.scene.env_origins
        initial_distance = sum(GOAL_DISTANCE_RANGE) / 2
        positions = origins + origins.new_tensor((0.0, -initial_distance, 0.005))
        env._mentorpi_goal_positions_w = positions
    return positions


def goal_offset_xy(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the world-aligned XY displacement from the block to its goal."""
    robot = cast(RigidObject, env.scene["robot"])
    return goal_positions_w(env)[:, :2] - robot.data.root_pos_w.torch[:, :2]


def lidar_ranges(env: ManagerBasedEnv) -> torch.Tensor:
    """Return one range in meters per ray, using max range for misses."""
    sensor: MultiMeshRayCaster = cast(MultiMeshRayCaster, env.scene.sensors["lidar"])
    scan: MultiMeshRayCasterData = sensor.data
    max_range = sensor.cfg.max_distance
    distances = torch.linalg.vector_norm(scan.ray_hits_w.torch - scan.pos_w.torch[:, None, :], dim=-1)
    return distances.nan_to_num(nan=max_range, posinf=max_range).clamp_max(max_range)


def goal_distance(env):
    return torch.linalg.vector_norm(goal_offset_xy(env), dim=-1)


def goal_distance_cost(env: ManagerBasedEnv) -> torch.Tensor:
    """Increase the cost with distance while bounding the maximum penalty."""
    return torch.expm1(goal_distance(env).clamp_max(2.5) / 2.0)


def velocity_toward_goal(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the signed closing speed toward each environment's goal."""
    offset = goal_offset_xy(env)
    direction = torch.nn.functional.normalize(offset, dim=-1, eps=1e-6)
    velocity_xy = mdp.root_lin_vel_w(env)[:, :2]
    return (velocity_xy * direction).sum(dim=-1)


def obstacle_clearance_cost(env: ManagerBasedEnv, margin: float) -> torch.Tensor:
    """Penalize motion that closes the gap to a nearby obstacle."""
    robot = cast(RigidObject, env.scene["robot"])
    robot_xy = robot.data.root_pos_w.torch[:, :2]
    obstacle_xy = torch.stack(
        [env.scene[name].data.root_pos_w.torch[:, :2] for name in ("obstacle_east", "obstacle_north", "obstacle_west")],
        dim=1,
    )
    offset = obstacle_xy - robot_xy[:, None, :]
    robot_size = cast(sim_utils.CuboidCfg, robot.cfg.spawn).size
    half_extents = 0.5 * (robot_xy.new_tensor(robot_size[:2]) + robot_xy.new_tensor(OBSTACLE_SIZE[:2]))
    # Both footprints are axis-aligned. The positive axis gaps form the shortest separation vector.
    axis_gap = offset.abs() - half_extents
    separation = offset.sign() * axis_gap.clamp_min(0.0)
    clearance = torch.linalg.vector_norm(separation, dim=-1)
    direction = torch.nn.functional.normalize(separation, dim=-1, eps=1e-6)
    # At contact or overlap, use the normal of the nearest face.
    contact_axis = axis_gap.argmax(dim=-1, keepdim=True)
    contact_direction = torch.zeros_like(offset).scatter_(-1, contact_axis, offset.sign().gather(-1, contact_axis))
    direction = torch.where(clearance[..., None] > 0.0, direction, contact_direction)
    closing_speed = (robot.data.root_vel_w.torch[:, None, :2] * direction).sum(dim=-1).clamp_min(0.0)
    proximity = (1.0 - clearance / margin).clamp(0.0, 1.0)
    ramp = proximity.square() * (3.0 - 2.0 * proximity)
    return (ramp * closing_speed).amax(dim=1)


def outside_arena(env: ManagerBasedEnv) -> torch.Tensor:
    position_xy = mdp.root_pos_w(env)[:, :2]
    return (position_xy.abs() > ARENA_HALF_WIDTH).any(dim=-1)


def goal_reached(env: ManagerBasedEnv) -> torch.Tensor:
    """Return whether the block's center is inside the square goal marker."""
    return (goal_offset_xy(env).abs() <= GOAL_HALF_WIDTH).all(dim=-1)


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        position = ObsTerm(func=mdp.root_pos_w)
        linear_velocity = ObsTerm(func=mdp.root_lin_vel_w)
        goal_offset = ObsTerm(func=goal_offset_xy)
        lidar = ObsTerm(func=lidar_ranges)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def reset_goal(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Sample a goal uniformly over the ring outside the obstacles."""
    marker = cast(VisualizationMarkers, env.scene["goal"])
    robot = cast(RigidObject, env.scene["robot"])
    positions = goal_positions_w(env)
    robot_xy = robot.data.root_pos_w.torch[env_ids, :2]
    angle = torch.empty_like(robot_xy[:, 0]).uniform_(0.0, 2 * torch.pi)
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    min_distance, max_distance = GOAL_DISTANCE_RANGE
    distance = torch.empty_like(robot_xy[:, :1]).uniform_(min_distance**2, max_distance**2).sqrt()
    positions[env_ids, :2] = robot_xy + direction * distance
    marker.visualize(translations=positions)


def reset_obstacles(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Place obstacles around the robot at independently sampled distances."""
    robot = cast(RigidObject, env.scene["robot"])
    robot_xy = robot.data.root_pos_w.torch[env_ids, :2]
    angle = torch.empty_like(robot_xy[:, 0]).uniform_(0.0, 2 * torch.pi)
    forward = torch.stack((angle.cos(), angle.sin()), dim=-1)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    origins = env.scene.env_origins[env_ids]
    for name, direction in (
        ("obstacle_east", forward),
        ("obstacle_north", lateral),
        ("obstacle_west", -lateral),
    ):
        obstacle = cast(RigidObject, env.scene[name])
        pose = obstacle.data.default_root_pose.torch[env_ids].clone()
        distance = torch.empty_like(robot_xy[:, :1]).uniform_(*OBSTACLE_DISTANCE_RANGE)
        pose[:, :2] = robot_xy + direction * distance
        pose[:, 2] += origins[:, 2]
        obstacle.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_goal = EventTerm(func=reset_goal, mode="reset")
    randomize_obstacles = EventTerm(func=reset_obstacles, mode="reset")


@configclass
class RewardsCfg:
    progress = RewTerm(func=velocity_toward_goal, weight=2.0)
    distance_to_goal = RewTerm(func=goal_distance_cost, weight=-0.1)
    obstacle_clearance = RewTerm(func=obstacle_clearance_cost, weight=-2.0, params={"margin": 0.10})
    reached_goal = RewTerm(func=goal_reached, weight=60.0)
    failure = RewTerm(
        func=mdp.is_terminated_term,
        weight=-120.0,
        params={"term_keys": ["collision", "out_of_bounds"]},
    )


def is_contacting_obstacle(env, threshold: float = 0.1) -> torch.Tensor:
    sensor = env.scene.sensors["obstacle_contacts"]
    forces = sensor.data.normal_force_matrix_w_history
    assert forces is not None, "Obstacle contact filters must be configured."

    magnitudes = torch.linalg.vector_norm(forces.torch, dim=-1)
    return (magnitudes > threshold).flatten(start_dim=1).any(dim=1)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached_goal = DoneTerm(func=goal_reached)

    collision = DoneTerm(func=is_contacting_obstacle)
    out_of_bounds = DoneTerm(func=outside_arena)


##
# Environment configuration
##


@configclass
class BalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cart-pole environment."""

    # Scene settings
    scene: MotionSceneCfg  = MotionSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 200
        # visualizer camera settings
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(8.0, 0.0, 5.0))
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # Isaac Lab's task loader resolves the preset to a backend config.
        self.sim.physics = MotionPhysicsCfg()  # pyright: ignore[reportAttributeAccessIssue]


def reset_iteration2_goal(env: ManagerBasedEnv, env_ids: torch.Tensor | slice) -> None:
    """Place the goal beyond a randomly selected side of the robot."""
    positions = goal_positions_w(env)
    sides = positions.new_tensor(((1, 0), (0, 1), (-1, 0), (0, -1)))
    origins_xy = env.scene.env_origins[env_ids, :2]
    side = torch.randint(len(sides), (origins_xy.shape[0],), device=positions.device)
    positions[env_ids, :2] = origins_xy + 1.5 * sides[side]
    cast(VisualizationMarkers, env.scene["goal"]).visualize(translations=positions)


def iteration2_obstacle_reset(name: str) -> EventTerm:
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )


@configclass
class Iteration2EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_obstacle_east = iteration2_obstacle_reset("obstacle_east")
    randomize_obstacle_north = iteration2_obstacle_reset("obstacle_north")
    randomize_obstacle_west = iteration2_obstacle_reset("obstacle_west")
    randomize_goal = EventTerm(func=reset_iteration2_goal, mode="reset")


@configclass
class Iteration2EnvCfg(BalanceEnvCfg):
    """Playback environment reconstructed from the first reduced-speed run."""

    events: Iteration2EventsCfg = Iteration2EventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        for obstacle in (self.scene.obstacle_east, self.scene.obstacle_north, self.scene.obstacle_west):
            obstacle.spawn.size = (0.35, 0.35, 0.2)
            x, y, _ = obstacle.init_state.pos
            obstacle.init_state.pos = (x, y, 0.1)
        self.scene.lidar.offset.pos = (0.0, 0.0, 0.06)
        self.actions.planar_velocity.speed_scale = 0.1
        self.actions.planar_velocity.speed_limit = 0.1
        self.rewards.progress.weight = 1.0
        self.rewards.obstacle_clearance = None
