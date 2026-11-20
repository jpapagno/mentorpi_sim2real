"""Batched navigation model, independent of USD, rendering, and Isaac Lab.

Coordinates are metres/radians, +x forward, +y left, +yaw counterclockwise.
The disc robot follows piecewise-linear, swept-checked segments. Optional
mecanum_v1 adds calibrated channel kinematics and configurable motor response;
it is still an analytic model without contact/roller physics. All obstacle
geometry or simulated first-return LiDAR is provided to the policy, by scenario.
"""

from dataclasses import dataclass
from collections import deque
import math
import random

import torch
from .drivetrain import DriveConfig


@dataclass(frozen=True)
class NavigationCfg:
    max_obstacles: int = 8
    obstacle_count: int = 4
    arena_half_extent: float = 2.0
    robot_radius: float = 0.15
    robot_size: tuple[float, float] | None = None
    policy_input: str = 'occupancy'
    reward_mode: str = 'progress'
    goal_normalization_half_extent: float | None = None
    obstacle_radius: float = 0.22
    speed_limits: tuple[float, float, float] = (0.2, 0.2, 0.5)
    acceleration_limits: tuple[float, float, float] = (0.5, 0.5, 1.5)
    dt: float = 0.05
    goal_tolerance: float = 0.15
    grid_resolution: float = 0.2
    planning_margin: float = 0.04
    occupancy_size: int = 100
    occupancy_resolution: float = 0.05
    occupancy_inflation: float = 0.0
    occupancy_source: str = 'geometry'
    lidar_rays: int = 500
    lidar_origin: tuple[float, float] = (-.012242, -.00008533)
    lidar_range_min: float = .02
    lidar_range_max: float = 25.
    lidar_scan_period: float = .1
    lidar_noise_std: float = 0.
    lidar_dropout: float = 0.
    lidar_noise_std_range: tuple[float, float] | None = None
    lidar_dropout_range: tuple[float, float] | None = None
    lidar_dropout_sectors: int = 0
    lidar_dropout_sector_degrees: tuple[float, float] = (1., 6.)
    lidar_outlier_probability: float = 0.
    lidar_outlier_error: float = .08
    start_obstacle_distance_range: tuple[float, float] | None = None
    min_obstacles: int | None = None
    obstacle_radius_range: tuple[float, float] | None = None
    min_goal_distance: float = 1.0
    layout_mode: str = "random"
    episode_seconds: float = 40.0
    progress_reward_scale: float = 5.0
    success_reward: float = 10.0
    collision_penalty: float = 10.0
    clearance_penalty: float = 0.0
    clearance_distance: float = 0.12
    actuator_scale_range: tuple[float, float] | None = None
    action_delay_steps: int = 0
    drive_model: str = 'body'
    wheel_diameter: float = .065
    wheelbase: float = .1368
    track_width: float = .1446
    max_wheel_rps: float = .8
    wheel_acceleration_rps2: float = 2.
    wheel_deadband_rps: float = 0.
    wheel_gains: tuple[float, float, float, float] = (1., 1., 1., 1.)
    command_gain: float = 1.
    command_limits: tuple[float,float,float] | None = None
    yaw_response: dict | None = None
    translation_only: bool = False
    translation_response: dict | None = None
    scene_spec: dict | None = None
    max_polygons: int = 16
    max_polygon_vertices: int = 8

    def __post_init__(self):
        if self.translation_response is not None:
            from .translation_response import validate_translation_response
            validate_translation_response(self.translation_response)
        if self.command_limits is not None and (len(self.command_limits)!=3 or any(not math.isfinite(v) or v<=0 or v>s for v,s in zip(self.command_limits,self.speed_limits))):
            raise ValueError('Invalid backend command limits')
        if self.yaw_response is not None:
            from .drivetrain_response import YawResponseCfg
            YawResponseCfg(**self.yaw_response)
            if self.command_limits is None or self.command_limits[2]>.3:
                raise ValueError('Measured yaw model requires yaw command cap<=0.3rad/s')
        if self.policy_input not in ('occupancy','laser_sectors') or self.reward_mode not in ('progress','td3_mecanum'):
            raise ValueError('Unknown policy input or reward mode')
        if self.policy_input=='laser_sectors' and self.occupancy_source!='lidar':
            raise ValueError('Sector policy requires current LiDAR sensing')
        if self.robot_size is not None:
            if len(self.robot_size)!=2 or any(not math.isfinite(v) or v<=0 for v in self.robot_size):
                raise ValueError('Robot rectangle needs positive finite length and width')
            if self.robot_radius < math.hypot(*self.robot_size)/2-1e-8:
                raise ValueError('Planning radius must enclose the rectangle')
        if self.goal_normalization_half_extent is not None and not 0 < self.goal_normalization_half_extent < math.inf:
            raise ValueError('Invalid goal normalization extent')
        if self.max_polygons < 1 or self.max_polygon_vertices < 3:
            raise ValueError('Invalid polygon capacity')
        if self.occupancy_source not in ('geometry', 'lidar'):
            raise ValueError('Unknown occupancy source')
        if self.occupancy_source == 'lidar':
            if self.occupancy_resolution != .05 or self.occupancy_inflation != 0:
                raise ValueError('LiDAR grid matches runtime: 0.05 m/cell and no geometry inflation')
            if not isinstance(self.lidar_rays, int) or self.lidar_rays < 20:
                raise ValueError('LiDAR needs at least 20 rays')
            if len(self.lidar_origin) != 2 or not all(math.isfinite(v) for v in self.lidar_origin):
                raise ValueError('LiDAR origin must be finite xy')
            if math.hypot(*self.lidar_origin) >= self.robot_radius:
                raise ValueError('LiDAR origin must lie inside robot footprint')
            if not (0 < self.lidar_range_min < self.lidar_range_max < math.inf):
                raise ValueError('Invalid LiDAR range limits')
            if self.dt <= 0 or not math.isfinite(self.lidar_scan_period) or self.lidar_scan_period < self.dt:
                raise ValueError('LiDAR scan period must be at least one simulation step')
            if not math.isclose(self.lidar_scan_period / self.dt, round(self.lidar_scan_period / self.dt)):
                raise ValueError('LiDAR scan period must be an integer number of simulation steps')
            if not 0 <= self.lidar_noise_std < math.inf or not 0 <= self.lidar_dropout <= 1:
                raise ValueError('Invalid LiDAR noise/dropout')
            for bounds, ceiling in ((self.lidar_noise_std_range, math.inf), (self.lidar_dropout_range, 1.)):
                if bounds is not None and (len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= ceiling
                                           or not all(math.isfinite(v) for v in bounds)):
                    raise ValueError('Invalid episode LiDAR randomization range')
            if not isinstance(self.lidar_dropout_sectors, int) or self.lidar_dropout_sectors < 0:
                raise ValueError('LiDAR sector count must be a nonnegative integer')
            bounds = self.lidar_dropout_sector_degrees
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] <= 180:
                raise ValueError('Invalid LiDAR dropout sector width')
            if not 0 <= self.lidar_outlier_probability <= 1 or not 0 <= self.lidar_outlier_error < math.inf:
                raise ValueError('Invalid LiDAR outlier settings')
        if self.drive_model not in ('body', 'mecanum_v1'):
            raise ValueError('Unknown drivetrain model')
        self.drivetrain_config()
        if not 0 <= self.obstacle_count <= self.max_obstacles:
            raise ValueError("obstacle_count must be between zero and max_obstacles")
        if self.dt <= 0 or self.robot_radius <= 0 or self.obstacle_radius <= 0:
            raise ValueError("Time step and radii must be positive")
        if self.arena_half_extent <= self.robot_radius + self.grid_resolution:
            raise ValueError("Arena is too small for the robot")
        if self.grid_resolution <= 0 or self.goal_tolerance <= 0 or self.planning_margin < 0:
            raise ValueError("Invalid planning or goal configuration")
        if any(v <= 0 for v in (*self.speed_limits, *self.acceleration_limits)):
            raise ValueError("Speed and acceleration limits must be positive")
        if self.occupancy_size != 100 or self.occupancy_resolution <= 0 or self.occupancy_inflation < 0:
            raise ValueError("Occupancy grid must be 100x100 with positive resolution and nonnegative inflation")
        if self.min_obstacles is not None and not 0 <= self.min_obstacles <= self.obstacle_count:
            raise ValueError("min_obstacles must lie between zero and obstacle_count")
        if self.obstacle_radius_range is not None and not 0 < self.obstacle_radius_range[0] <= self.obstacle_radius_range[1]:
            raise ValueError("Invalid obstacle_radius_range")
        if self.min_goal_distance <= 0 or self.episode_seconds <= 0 or self.clearance_distance <= 0:
            raise ValueError("Distances and episode duration must be positive")
        if self.layout_mode not in {"random", "passage", "mixed", "shapes", "clutter"}:
            raise ValueError("Unknown layout_mode")
        if self.start_obstacle_distance_range is not None:
            bounds = self.start_obstacle_distance_range
            if len(bounds) != 2 or not self.robot_radius < bounds[0] <= bounds[1] < math.inf:
                raise ValueError('Start obstacle distances must exceed robot radius')
        if self.action_delay_steps < 0:
            raise ValueError("action_delay_steps must be nonnegative")
        if self.actuator_scale_range is not None and not 0 < self.actuator_scale_range[0] <= self.actuator_scale_range[1] <= 1:
            raise ValueError("Actuator scale range must lie in (0,1]")
        if self.layout_mode != "random" and self.max_obstacles < 6:
            raise ValueError("Passage scenarios require capacity for at least six obstacles")

    @property
    def observation_size(self):
        # Goal xy, body velocity, last action; occupancy is a separate CNN input.
        return 8

    def drivetrain_config(self):
        values = {name: getattr(self,name) for name in DriveConfig.__dataclass_fields__ if name != 'body_acceleration_limits'}
        return DriveConfig(**values,body_acceleration_limits=self.acceleration_limits)


def world_to_body(delta: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 2) vectors into a body's frame (yaw broadcast-compatible)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * delta[..., 0] + s * delta[..., 1],
                        -s * delta[..., 0] + c * delta[..., 1]), dim=-1)


def swept_collision(start, end, obstacles, valid, robot_radius, arena_half_extent):
    """Disc versus circles along the entire segment, plus the square arena boundary."""
    segment = end - start
    to_obstacle = obstacles[..., :2] - start[:, None, :]
    t = (to_obstacle * segment[:, None, :]).sum(-1)
    t = (t / segment.square().sum(-1, keepdim=True).clamp_min(1e-12)).clamp(0, 1)
    nearest = start[:, None, :] + t[..., None] * segment[:, None, :]
    clearance = (obstacles[..., :2] - nearest).square().sum(-1)
    hit = (clearance <= (obstacles[..., 2] + robot_radius).square()) & valid
    boundary = end.abs().amax(-1) >= arena_half_extent - robot_radius
    return hit.any(-1) | boundary


def sample_layout(cfg: NavigationCfg, rng: random.Random):
    """CPU reset sampler. Four-connected free grid paths guarantee a route exists.

    Obstacles are inflated by robot radius, margin, and half a cell diagonal;
    therefore an edge between free adjacent cell centres is also collision-free.
    Starts/goals are cell centres in the same connected component. This sampler
    prioritizes verifiable resets over massive-parallel-reset throughput.
    """
    h, cell = cfg.arena_half_extent, cfg.grid_resolution
    limit = h - cfg.robot_radius - cfg.planning_margin
    n = int(2 * limit / cell) + 1
    axis = [(i - (n - 1) / 2) * cell for i in range(n)]
    for _ in range(100):
        passage = cfg.layout_mode == "passage" or (cfg.layout_mode == "mixed" and rng.random() < .25)
        if passage:
            # A touching chain of circles forms a divider with one central
            # opening; starts/goals must lie on opposing sides of the divider.
            gap = rng.uniform(.55, .75)
            radius = max(.30, (h - gap/2) / cfg.max_obstacles)
            per_side = math.ceil((h - gap/2) / (2*radius))
            if 2*per_side > cfg.max_obstacles:
                continue
            obstacles = [(0.0, sign*(gap/2 + radius + 2*radius*i), radius)
                         for sign in (-1, 1) for i in range(per_side)]
        else:
            count = cfg.obstacle_count if cfg.min_obstacles is None else rng.randint(cfg.min_obstacles, cfg.obstacle_count)
            obstacles = []
            if cfg.layout_mode == "clutter":
                # Thin posts, uneven short furniture faces, and compound patches.
                # All pieces remain real collision geometry, not image artifacts.
                while len(obstacles) < count:
                    kind = rng.choices(('post', 'face', 'patch'), weights=(.45, .35, .20))[0]
                    length = min(1 if kind == 'post' else rng.randint(2, 7), count-len(obstacles))
                    radius = rng.uniform(.025, .065) if kind == 'post' else rng.uniform(.07, .16)
                    cx, cy = rng.uniform(-h+.35, h-.35), rng.uniform(-h+.35, h-.35)
                    angle = rng.uniform(-math.pi, math.pi)
                    spacing = radius*rng.uniform(1.2, 1.6)
                    for index in range(length):
                        if kind == 'patch':
                            x, y = (index % 3-1)*spacing, (index//3-.5)*spacing
                        else:
                            x, y = (index-(length-1)/2)*spacing, rng.uniform(-.3, .3)*radius
                        r = radius*rng.uniform(.8, 1.2)
                        obstacles.append((cx+math.cos(angle)*x-math.sin(angle)*y,
                                          cy+math.sin(angle)*x+math.cos(angle)*y, r))
            elif cfg.layout_mode == "shapes":
                # Compound footprints: segments, L bends and filled patches.
                # Their union drives both exact disc sweeps and map generation.
                while len(obstacles) < count:
                    length = min(rng.randint(2,6),count-len(obstacles))
                    cx,cy=rng.uniform(-h+.7,h-.7),rng.uniform(-h+.7,h-.7)
                    angle=rng.uniform(-math.pi,math.pi)
                    radius=rng.uniform(.10,.18)
                    shape=rng.choice(('segment','elbow','patch'))
                    for index in range(length):
                        if shape=='patch':x,y=(index%3-1)*radius*1.4,(index//3-.5)*radius*1.4
                        elif shape=='elbow':x,y=min(index,length//2)*radius*1.4,max(0,index-length//2)*radius*1.4
                        else:x,y=(index-(length-1)/2)*radius*1.4,0.
                        obstacles.append((cx+math.cos(angle)*x-math.sin(angle)*y,
                                          cy+math.sin(angle)*x+math.cos(angle)*y,radius))
            else:
                for _ in range(count):
                    x, y = rng.uniform(-h + .5, h - .5), rng.uniform(-h + .5, h - .5)
                    radius = cfg.obstacle_radius if cfg.obstacle_radius_range is None else rng.uniform(*cfg.obstacle_radius_range)
                    obstacles.append((x, y, radius))
        inflation = cfg.robot_radius + cfg.planning_margin + cell / math.sqrt(2)
        free = {(i, j) for i in range(n) for j in range(n)
                if all(math.hypot(axis[i] - x, axis[j] - y) > r + inflation
                       for x, y, r in obstacles)}
        while free:
            first = min(free)
            free.remove(first)
            component, queue = [first], deque([first])
            while queue:
                i, j = queue.popleft()
                for other in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if other in free:
                        free.remove(other)
                        component.append(other)
                        queue.append(other)
            rng.shuffle(component)
            if len(component) < 2:
                continue
            starts = [p for p in component if not passage or abs(axis[p[0]]) > .6]
            if cfg.start_obstacle_distance_range is not None:
                low, high = cfg.start_obstacle_distance_range
                starts = [p for p in starts if low <= min(
                    (math.hypot(axis[p[0]]-x, axis[p[1]]-y)-r for x, y, r in obstacles),
                    default=math.inf) <= high]
            if not starts:
                continue
            start = starts[0]
            goals = [p for p in component if math.hypot(
                axis[p[0]] - axis[start[0]], axis[p[1]] - axis[start[1]]) >= cfg.min_goal_distance
                and (not passage or axis[p[0]] * axis[start[0]] < 0 and abs(axis[p[0]]) > .6)]
            if goals:
                goal = rng.choice(goals)
                return ([axis[start[0]], axis[start[1]]],
                        [axis[goal[0]], axis[goal[1]]], obstacles, rng.uniform(-math.pi, math.pi))
    raise RuntimeError("Could not sample a reachable layout; reduce obstacle density")


class NavigationState:
    def __init__(self, num_envs, cfg=None, device="cpu", seed=42):
        self.cfg = cfg or NavigationCfg()
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.num_envs = num_envs
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.xy = torch.zeros(num_envs, 2, device=device)
        self.yaw = torch.zeros(num_envs, device=device)
        self.goal = torch.zeros_like(self.xy)
        self.velocity = torch.zeros(num_envs, 3, device=device)
        self.actions = torch.zeros_like(self.velocity)
        self.obstacles = torch.zeros(num_envs, self.cfg.max_obstacles, 3, device=device)
        self.valid = torch.zeros(num_envs, self.cfg.max_obstacles, dtype=torch.bool, device=device)
        self.polygons = torch.zeros(num_envs,self.cfg.max_polygons,self.cfg.max_polygon_vertices,2,device=device)
        self.polygon_counts = torch.zeros(num_envs,self.cfg.max_polygons,dtype=torch.long,device=device)
        self.polygon_slots = 0
        self.collision = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.collision)
        self.reward = torch.zeros(num_envs, device=device)
        self.limits = torch.tensor(self.cfg.speed_limits, device=device)
        self.acceleration = torch.tensor(self.cfg.acceleration_limits, device=device)
        self.actuator_scale = torch.ones(num_envs,3,device=device)
        self.delayed_actions = torch.zeros(num_envs,max(1,self.cfg.action_delay_steps),3,device=device)
        self.drive = self.cfg.drivetrain_config()
        self.wheel_rps = torch.zeros(num_envs,4,device=device)
        self.wheel_command_rps = torch.zeros_like(self.wheel_rps)
        self.body_command_velocity = torch.zeros_like(self.velocity)
        self.translation_response=None
        if self.cfg.translation_response is not None and self.cfg.translation_response['version']=='translation_history_v1':
            from .translation_response import BatchedTranslationResponse
            self.translation_response=BatchedTranslationResponse(num_envs,device,self.cfg.translation_response)
        self.yaw_response=None
        if self.cfg.yaw_response is not None:
            from .drivetrain_response import BatchedYawResponse,YawResponseCfg
            self.yaw_response=BatchedYawResponse(num_envs,device,YawResponseCfg(**self.cfg.yaw_response))
        self._wheel_inverse = torch.tensor(self.drive.inverse_matrix,device=device)
        self._wheel_forward = torch.tensor(self.drive.forward_matrix,device=device)
        self._wheel_gains = torch.tensor(self.drive.wheel_gains,device=device)
        axis = ((self.cfg.occupancy_size - 1) / 2 - torch.arange(
            self.cfg.occupancy_size, device=device)) * self.cfg.occupancy_resolution
        forward, left = torch.meshgrid(axis, axis, indexing="ij")
        self._grid_cells = torch.stack((forward, left), dim=-1)
        self.lidar = None
        if self.cfg.occupancy_source == 'lidar':
            from .lidar import PlanarLidar
            self.lidar = PlanarLidar(num_envs, self.cfg, self.device, seed)

    def reset(self, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else torch.as_tensor(env_ids).cpu().tolist()
        if not ids:
            return
        if self.cfg.scene_spec is None:
            layouts = [(*sample_layout(self.cfg,self.rng),[]) for _ in ids]
        else:
            from .scenes import sample_scene
            layouts = [sample_scene(self.cfg,self.rng) for _ in ids]
        self.polygons[ids] = 0
        self.polygon_counts[ids] = 0
        for env, layout in zip(ids,layouts):
            for slot, vertices in enumerate(layout[4]):
                self.polygons[env,slot,:len(vertices)] = torch.tensor(vertices,device=self.device)
                self.polygon_counts[env,slot] = len(vertices)
        self.polygon_slots = int((self.polygon_counts>0).sum(-1).max())
        self.xy[ids] = torch.tensor([p[0] for p in layouts], device=self.device)
        self.goal[ids] = torch.tensor([p[1] for p in layouts], device=self.device)
        self.yaw[ids] = torch.tensor([p[3] for p in layouts], device=self.device)
        self.obstacles[ids] = 0
        self.valid[ids] = False
        padded = [p[2] + [(0., 0., 0.)] * (self.cfg.max_obstacles - len(p[2])) for p in layouts]
        self.obstacles[ids] = torch.tensor(padded, device=self.device)
        counts = torch.tensor([len(p[2]) for p in layouts], device=self.device)
        self.valid[ids] = torch.arange(self.cfg.max_obstacles, device=self.device)[None, :] < counts[:, None]
        self.velocity[ids] = 0
        self.actions[ids] = 0
        self.collision[ids] = False
        self.success[ids] = False
        self.reward[ids] = 0
        self.delayed_actions[ids] = 0
        self.wheel_rps[ids] = 0
        self.wheel_command_rps[ids] = 0
        self.body_command_velocity[ids] = 0
        if self.yaw_response is not None:self.yaw_response.reset(ids)
        if self.translation_response is not None:self.translation_response.reset(ids)
        if self.lidar is not None:
            self.lidar.reset(ids)
        if self.cfg.actuator_scale_range is not None:
            self.actuator_scale[ids] = torch.tensor([[self.rng.uniform(*self.cfg.actuator_scale_range) for _ in range(3)] for _ in ids],device=self.device)

    def step(self, actions):
        if actions.shape != (self.num_envs, 3) or not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite with shape (num_envs, 3)")
        cfg = self.cfg
        actions = actions.to(self.device).clamp(-1, 1)
        if cfg.translation_only:
            actions=actions.clone();actions[:,2]=0
        old_distance = torch.linalg.vector_norm(self.goal - self.xy, dim=-1)
        applied = actions
        if cfg.action_delay_steps:
            applied=self.delayed_actions[:,0].clone()
            self.delayed_actions[:,:-1]=self.delayed_actions[:,1:].clone()
            self.delayed_actions[:,-1]=actions
        response_active=self.yaw_response is not None or cfg.translation_response is not None
        previous_velocity = self.body_command_velocity if cfg.drive_model == 'mecanum_v1' or response_active else self.velocity
        target_velocity=applied*self.limits*self.actuator_scale
        if cfg.command_limits is not None:
            caps=target_velocity.new_tensor(cfg.command_limits)
            target_velocity=target_velocity.clamp(-caps,caps)
        difference = target_velocity - previous_velocity
        max_change = self.acceleration * cfg.dt
        velocity = previous_velocity + difference.clamp(-max_change, max_change)
        if cfg.drive_model=='mecanum_v1' or response_active:self.body_command_velocity=velocity
        if cfg.drive_model == 'mecanum_v1':
            targets = self.drive.command_gain*(velocity @ self._wheel_inverse.T)
            ratio = (targets.abs().amax(-1,keepdim=True)/self.drive.max_wheel_rps).clamp_min(1.)
            self.wheel_command_rps = targets/ratio
            wheel_delta = self.drive.wheel_acceleration_rps2*cfg.dt
            self.wheel_rps += (self.wheel_command_rps-self.wheel_rps).clamp(-wheel_delta,wheel_delta)
            realized = torch.where(self.wheel_rps.abs() < self.drive.wheel_deadband_rps,
                                   0.,self.wheel_rps*self._wheel_gains)
            velocity = realized @ self._wheel_forward.T
        if cfg.translation_response is not None:
            from .translation_response import translation_response_torch
            velocity=self.translation_response.step(velocity,cfg.dt) if self.translation_response is not None else translation_response_torch(velocity,cfg.translation_response)
        if self.yaw_response is not None:
            velocity=velocity.clone()
            velocity[:,2]=self.yaw_response.step(velocity[:,2],cfg.dt)
        midpoint_yaw = self.yaw + velocity[:, 2] * cfg.dt / 2
        # Inverse rotation: transform the commanded body displacement to world.
        proposed = self.xy + world_to_body(velocity[:, :2] * cfg.dt, -midpoint_yaw)
        proposed_yaw=self.yaw+velocity[:,2]*cfg.dt
        if cfg.robot_size is not None:
            from .geometry import rectangle_sweep
            self.collision=rectangle_sweep(self.xy,proposed,self.yaw,proposed_yaw,cfg.robot_size,
                self.obstacles,self.valid,self.polygons[:,:self.polygon_slots],self.polygon_counts[:,:self.polygon_slots],cfg.arena_half_extent)
        else:
            self.collision = swept_collision(self.xy, proposed, self.obstacles, self.valid,
                                             cfg.robot_radius, cfg.arena_half_extent)
        if cfg.scene_spec is not None and cfg.robot_size is None:
            from .geometry import polygon_sweep
            self.collision |= polygon_sweep(self.xy,proposed,self.polygons[:,:self.polygon_slots],self.polygon_counts[:,:self.polygon_slots],cfg.robot_radius)
        self.xy = torch.where(self.collision[:, None], self.xy, proposed)
        self.velocity = torch.where(self.collision[:, None], 0.0, velocity)
        self.wheel_rps[self.collision] = 0
        self.wheel_command_rps[self.collision] = 0
        self.body_command_velocity[self.collision] = 0
        self.yaw = (self.yaw + self.velocity[:, 2] * cfg.dt + math.pi) % (2 * math.pi) - math.pi
        distance = torch.linalg.vector_norm(self.goal - self.xy, dim=-1)
        self.success = (distance < cfg.goal_tolerance) & ~self.collision
        clearance = self.minimum_clearance()
        proximity_cost = (1 - clearance / cfg.clearance_distance).clamp(0, 1)
        self.reward = (cfg.progress_reward_scale * (old_distance - distance) - 0.01
                       - 0.01 * (actions - self.actions).square().sum(-1)
                       + cfg.success_reward * self.success.float() - cfg.collision_penalty * self.collision.float()
                       - cfg.clearance_penalty * proximity_cost)
        if cfg.reward_mode=='td3_mecanum':
            relative=world_to_body(self.goal-self.xy,self.yaw)
            unit=relative/relative.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            toward=(self.velocity[:,:2]*unit).sum(-1)/max(cfg.speed_limits[:2])
            self.reward=.5*toward.clamp(-1,1)-.5*(self.velocity[:,2]/cfg.speed_limits[2]).abs()-.5*proximity_cost
            self.reward=torch.where(self.success,cfg.success_reward,self.reward)
            self.reward=torch.where(self.collision,-cfg.collision_penalty,self.reward)
        self.actions = actions.clone()
        if self.lidar is not None:
            self.lidar.advance()
        return self.reward

    def minimum_clearance(self):
        if self.cfg.robot_size is not None:
            from .geometry import rectangle_clearance
            return rectangle_clearance(self.xy,self.yaw,self.cfg.robot_size,self.obstacles,self.valid,
                self.polygons[:,:self.polygon_slots],self.polygon_counts[:,:self.polygon_slots],self.cfg.arena_half_extent)
        distance = torch.linalg.vector_norm(self.obstacles[..., :2] - self.xy[:, None], dim=-1)
        obstacle = (distance - self.obstacles[..., 2] - self.cfg.robot_radius).masked_fill(~self.valid, float("inf"))
        walls = self.cfg.arena_half_extent - self.xy.abs().amax(-1) - self.cfg.robot_radius
        clearance = torch.minimum(obstacle.amin(-1), walls)
        if self.cfg.scene_spec is not None:
            from .geometry import polygon_distance
            clearance = torch.minimum(clearance,polygon_distance(self.xy[:,None],self.polygons[:,:self.polygon_slots],self.polygon_counts[:,:self.polygon_slots])[:,0]-self.cfg.robot_radius)
        return clearance

    def occupancy_grid(self):
        """Binary local map [env, 1, row, col], 0 free / 1 occupied.

        Top is robot-forward, left is robot-left. Robot lies at the intersection
        of the middle four cells. Circle/tile intersection is exact in this
        body-aligned frame; any tile crossing a wall is marked occupied.
        Generation uses the same geometry/poses as task collision and visuals.
        The optional LiDAR source instead clears only observed rays; unknown
        stays blocked, using the real runtime's rasterization and scan cadence.
        """
        if self.lidar is not None:
            return self.lidar.observe(self.xy, self.yaw, self.obstacles, self.valid,self.polygons[:,:self.polygon_slots],self.polygon_counts[:,:self.polygon_slots])
        cfg = self.cfg
        cells = self._grid_cells
        half_cell = cfg.occupancy_resolution / 2
        c, s = self.yaw.cos()[:, None, None], self.yaw.sin()[:, None, None]
        world_x = self.xy[:, 0, None, None] + c * cells[..., 0] - s * cells[..., 1]
        world_y = self.xy[:, 1, None, None] + s * cells[..., 0] + c * cells[..., 1]
        # Projection of the rotated tile half-size onto each world axis.
        projection = half_cell * (c.abs() + s.abs())
        wall_limit = cfg.arena_half_extent - cfg.occupancy_inflation
        # Conservative micrometre tolerance keeps tangent tiles stable under
        # floating-point roundoff (notably quarter-turn rotations).
        occupied = (world_x.abs() + projection >= wall_limit - 1e-6) | (world_y.abs() + projection >= wall_limit - 1e-6)
        relative = world_to_body(self.obstacles[..., :2] - self.xy[:, None, :], self.yaw[:, None])
        # Iterate slots, vectorizing pixels/environments; avoids a large BxKxHxWx2 allocation.
        for i in range(cfg.max_obstacles):
            dx = (cells[..., 0] - relative[:, i, 0, None, None]).abs().sub(half_cell).clamp_min(0)
            dy = (cells[..., 1] - relative[:, i, 1, None, None]).abs().sub(half_cell).clamp_min(0)
            radius = self.obstacles[:, i, 2, None, None] + cfg.occupancy_inflation
            occupied |= (dx.square() + dy.square() <= radius.square()) & self.valid[:, i, None, None]
        if cfg.scene_spec is not None:
            from .geometry import polygon_distance
            points = torch.stack((world_x,world_y),-1).reshape(self.num_envs,-1,2)
            # Conservative polygon debug raster; LiDAR uses exact edge hits.
            occupied |= polygon_distance(points,self.polygons[:,:self.polygon_slots],self.polygon_counts[:,:self.polygon_slots]).reshape_as(occupied) <= half_cell*math.sqrt(2)+cfg.occupancy_inflation
        return occupied[:, None].float()

    def observations(self):
        cfg = self.cfg
        if cfg.policy_input=='laser_sectors':
            from .robot.laser_sectors import sector_observation_torch
            self.lidar.observe(self.xy,self.yaw,self.obstacles,self.valid,self.polygons[:,:self.polygon_slots],
                               self.polygon_counts[:,:self.polygon_slots],rasterize=False)
            ranges=torch.minimum(self.lidar.ranges[:,self.lidar.lower],self.lidar.ranges[:,self.lidar.upper])
            points=ranges[...,None]*self.lidar.normalized_directions+self.lidar.origin
            return {'policy':sector_observation_torch(points,world_to_body(self.goal-self.xy,self.yaw),self.actions)}
        scale = 2 * math.sqrt(2) * (cfg.goal_normalization_half_extent or cfg.arena_half_extent)
        state = torch.cat((world_to_body(self.goal - self.xy, self.yaw) / scale,
                           self.velocity / self.limits, self.actions), dim=-1)
        return {"policy": state, "occupancy": self.occupancy_grid()}
