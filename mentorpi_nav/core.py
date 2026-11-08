"""Batched navigation model, independent of USD, rendering, and Isaac Lab.

Coordinates are metres/radians, +x forward, +y left, +yaw counterclockwise.
The disc robot follows piecewise-linear, swept-checked segments. It is not a
wheel/traction simulator. All obstacle geometry is provided to the policy.
"""

from dataclasses import dataclass
from collections import deque
import math
import random

import torch


@dataclass(frozen=True)
class NavigationCfg:
    max_obstacles: int = 8
    obstacle_count: int = 4
    arena_half_extent: float = 2.0
    robot_radius: float = 0.15
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

    def __post_init__(self):
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
        if self.layout_mode not in {"random", "passage", "mixed", "shapes"}:
            raise ValueError("Unknown layout_mode")
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
            if cfg.layout_mode == "shapes":
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
        self.collision = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.collision)
        self.reward = torch.zeros(num_envs, device=device)
        self.limits = torch.tensor(self.cfg.speed_limits, device=device)
        self.acceleration = torch.tensor(self.cfg.acceleration_limits, device=device)
        self.actuator_scale = torch.ones(num_envs,3,device=device)
        self.delayed_actions = torch.zeros(num_envs,max(1,self.cfg.action_delay_steps),3,device=device)
        axis = ((self.cfg.occupancy_size - 1) / 2 - torch.arange(
            self.cfg.occupancy_size, device=device)) * self.cfg.occupancy_resolution
        forward, left = torch.meshgrid(axis, axis, indexing="ij")
        self._grid_cells = torch.stack((forward, left), dim=-1)

    def reset(self, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else torch.as_tensor(env_ids).cpu().tolist()
        if not ids:
            return
        layouts = [sample_layout(self.cfg, self.rng) for _ in ids]
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
        if self.cfg.actuator_scale_range is not None:
            self.actuator_scale[ids] = torch.tensor([[self.rng.uniform(*self.cfg.actuator_scale_range) for _ in range(3)] for _ in ids],device=self.device)

    def step(self, actions):
        if actions.shape != (self.num_envs, 3) or not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite with shape (num_envs, 3)")
        cfg = self.cfg
        actions = actions.to(self.device).clamp(-1, 1)
        old_distance = torch.linalg.vector_norm(self.goal - self.xy, dim=-1)
        applied = actions
        if cfg.action_delay_steps:
            applied=self.delayed_actions[:,0].clone()
            self.delayed_actions[:,:-1]=self.delayed_actions[:,1:].clone()
            self.delayed_actions[:,-1]=actions
        difference = applied * self.limits * self.actuator_scale - self.velocity
        max_change = self.acceleration * cfg.dt
        velocity = self.velocity + difference.clamp(-max_change, max_change)
        midpoint_yaw = self.yaw + velocity[:, 2] * cfg.dt / 2
        # Inverse rotation: transform the commanded body displacement to world.
        proposed = self.xy + world_to_body(velocity[:, :2] * cfg.dt, -midpoint_yaw)
        self.collision = swept_collision(self.xy, proposed, self.obstacles, self.valid,
                                         cfg.robot_radius, cfg.arena_half_extent)
        self.xy = torch.where(self.collision[:, None], self.xy, proposed)
        self.velocity = torch.where(self.collision[:, None], 0.0, velocity)
        self.yaw = (self.yaw + self.velocity[:, 2] * cfg.dt + math.pi) % (2 * math.pi) - math.pi
        distance = torch.linalg.vector_norm(self.goal - self.xy, dim=-1)
        self.success = (distance < cfg.goal_tolerance) & ~self.collision
        clearance = self.minimum_clearance()
        proximity_cost = (1 - clearance / cfg.clearance_distance).clamp(0, 1)
        self.reward = (cfg.progress_reward_scale * (old_distance - distance) - 0.01
                       - 0.01 * (actions - self.actions).square().sum(-1)
                       + cfg.success_reward * self.success.float() - cfg.collision_penalty * self.collision.float()
                       - cfg.clearance_penalty * proximity_cost)
        self.actions = actions.clone()
        return self.reward

    def minimum_clearance(self):
        distance = torch.linalg.vector_norm(self.obstacles[..., :2] - self.xy[:, None], dim=-1)
        obstacle = (distance - self.obstacles[..., 2] - self.cfg.robot_radius).masked_fill(~self.valid, float("inf"))
        walls = self.cfg.arena_half_extent - self.xy.abs().amax(-1) - self.cfg.robot_radius
        return torch.minimum(obstacle.amin(-1), walls)

    def occupancy_grid(self):
        """Binary local map [env, 1, row, col], 0 free / 1 occupied.

        Top is robot-forward, left is robot-left. Robot lies at the intersection
        of the middle four cells. Circle/tile intersection is exact in this
        body-aligned frame; any tile crossing a wall is marked occupied.
        Generation uses the same geometry/poses as task collision and visuals.
        No sensors, occlusion, unknown state, or robot-self occupancy are used.
        """
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
        return occupied[:, None].float()

    def observations(self):
        cfg = self.cfg
        scale = 2 * math.sqrt(2) * cfg.arena_half_extent
        state = torch.cat((world_to_body(self.goal - self.xy, self.yaw) / scale,
                           self.velocity / self.limits, self.actions), dim=-1)
        return {"policy": state, "occupancy": self.occupancy_grid()}
