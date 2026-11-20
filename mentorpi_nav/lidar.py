"""Analytic first-return LiDAR and the robot's ray-grid contract, in Torch.

This models scan visibility, not optical materials or contact physics. Unknown
cells remain blocked; finite endpoints get the runtime's one-cell uncertainty.
"""
import math

import torch


def scan_grid(points, origin):
    """Batched equivalent of robot.live_occupancy_from_scan, fixed 5 cm grid.

    points: [B, rays, 2] body coordinates; invalid rays contain NaNs.
    origin: [2] body coordinates. Returns [B, 1, 100, 100].
    """
    # Double precision matches NumPy rasterization at cell boundaries.
    points = points.double()
    origin = origin.double()
    batch, rays, _ = points.shape
    finite = torch.isfinite(points).all(-1)
    points = torch.where(finite[..., None], points, 0.)
    start = 50 - origin / .05
    ends = 50 - points / .05
    direction = ends - start
    scale = torch.ones_like(direction[..., 0])
    for axis in (0, 1):
        delta = direction[..., axis]
        safe = torch.where(delta != 0, delta, 1.)
        limit = torch.where(delta > 0, (99.999 - start[axis]) / safe,
                            torch.where(delta < 0, -start[axis] / safe, 1.))
        scale = torch.minimum(scale, limit)
    direction = direction * scale[..., None]
    lengths = direction.abs().amax(-1).ceil().long() + 1
    steps = torch.arange(102, device=points.device)
    cells = (start + direction[..., None, :] *
             (steps / lengths[..., None])[..., None]).floor().long()
    rr, cc = cells.unbind(-1)
    valid = finite[..., None] & (steps <= lengths[..., None]) & (rr >= 0) & (rr < 100) & (cc >= 0) & (cc < 100)
    index = (rr * 100 + cc).clamp(0, 9999).reshape(batch, -1)
    visible = torch.zeros(batch, 10000, device=points.device, dtype=torch.int32)
    visible.scatter_add_(1, index, valid.reshape(batch, -1).int())
    axis = (49.5 - torch.arange(100, device=points.device, dtype=torch.float64)) * .05
    footprint = axis[:, None].square() + axis[None, :].square() <= .15 ** 2
    blocked = ~(visible.reshape(batch, 100, 100).bool() | footprint)
    endpoint = ends.floor().long()
    occupied = torch.zeros_like(visible)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = endpoint[..., 0] + dr, endpoint[..., 1] + dc
            valid = finite & (rr >= 0) & (rr < 100) & (cc >= 0) & (cc < 100)
            occupied.scatter_add_(1, (rr * 100 + cc).clamp(0, 9999), valid.int())
    return (blocked | occupied.reshape(batch, 100, 100).bool())[:, None].float()


class PlanarLidar:
    """Seeded sensor state, refreshed at scan cadence and held between scans."""
    def __init__(self, batch, cfg, device, seed):
        self.cfg = cfg
        self.device = device
        self.origin = torch.tensor(cfg.lidar_origin, device=device, dtype=torch.float64)
        # Captured LD19 raw metadata includes both angular endpoints.
        self.angles = torch.linspace(0, 2 * math.pi, cfg.lidar_rays, device=device, dtype=torch.float64)
        self.directions = torch.stack((self.angles.cos(), self.angles.sin()), -1)
        target = -math.pi + torch.arange(720, device=device, dtype=torch.float64) * (2 * math.pi / 720)
        source = (target % (2 * math.pi)) / (2 * math.pi / (cfg.lidar_rays - 1))
        self.lower = source.floor().long() % cfg.lidar_rays
        self.upper = (self.lower + 1) % cfg.lidar_rays
        self.normalized_directions = torch.stack((target.cos(), target.sin()), -1)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.ticks = torch.zeros(batch, device=device, dtype=torch.long)
        self.sample_tick = torch.full_like(self.ticks, -1)
        self.grid_tick = torch.full_like(self.ticks, -2)
        self.period = round(cfg.lidar_scan_period / cfg.dt)
        self.grid = torch.ones(batch, 1, 100, 100, device=device)
        self.ranges = torch.full((batch, cfg.lidar_rays), float('inf'), device=device, dtype=torch.float64)
        self.noise_std = torch.full((batch, 1), cfg.lidar_noise_std, device=device, dtype=torch.float64)
        self.dropout = torch.full((batch, 1), cfg.lidar_dropout, device=device, dtype=torch.float64)
        self.sector_mask = torch.zeros(batch, cfg.lidar_rays, device=device, dtype=torch.bool)
        self.reset(torch.arange(batch, device=device))

    def reset(self, ids):
        self.ticks[ids] = 0
        self.sample_tick[ids] = -1
        self.grid_tick[ids] = -2
        cfg = self.cfg
        for target, bounds, default in ((self.noise_std, cfg.lidar_noise_std_range, cfg.lidar_noise_std),
                                        (self.dropout, cfg.lidar_dropout_range, cfg.lidar_dropout)):
            if bounds is None:
                target[ids] = default
            else:
                draw = torch.rand((len(ids), 1), device=self.device, dtype=target.dtype, generator=self.generator)
                target[ids] = bounds[0] + draw*(bounds[1]-bounds[0])
        self.sector_mask[ids] = False
        if cfg.lidar_dropout_sectors:
            shape = (len(ids), cfg.lidar_dropout_sectors, 1)
            centers = torch.rand(shape, device=self.device, generator=self.generator)*2*math.pi
            low, high = cfg.lidar_dropout_sector_degrees
            widths = (low + torch.rand(shape, device=self.device, generator=self.generator)*(high-low))*math.pi/180
            delta = (self.angles - centers + math.pi) % (2*math.pi) - math.pi
            self.sector_mask[ids] = (delta.abs() <= widths/2).any(1)

    def advance(self):
        self.ticks += 1

    def raycast(self, xy, yaw, obstacles, valid, polygons=None, polygon_counts=None):
        """Nearest circle-union or arena-wall intersection from sensor origin."""
        cfg = self.cfg
        c, s = yaw.double().cos(), yaw.double().sin()
        origin = xy.double() + torch.stack((c*self.origin[0]-s*self.origin[1],
                                           s*self.origin[0]+c*self.origin[1]), -1)
        directions = torch.stack((c[:, None]*self.directions[:, 0]-s[:, None]*self.directions[:, 1],
                                  s[:, None]*self.directions[:, 0]+c[:, None]*self.directions[:, 1]), -1)
        safe = torch.where(directions.abs() > 1e-12, directions, 1.)
        wall = (torch.where(directions >= 0, cfg.arena_half_extent, -cfg.arena_half_extent) - origin[:, None]) / safe
        wall = wall.masked_fill(directions.abs() <= 1e-12, float('inf'))
        ranges = wall.amin(-1).clamp_min(0)
        for slot in range(obstacles.shape[1]):
            delta = obstacles[:, slot, :2].double() - origin
            projection = (delta[:, None] * directions).sum(-1)
            disc = projection.square() - (delta.square().sum(-1) - obstacles[:, slot, 2].double().square())[:, None]
            root = disc.clamp_min(0).sqrt()
            near, far = projection - root, projection + root
            distance = torch.where(near >= 0, near, far)
            hit = valid[:, slot, None] & (disc >= 0) & (distance >= 0)
            ranges = torch.minimum(ranges, torch.where(hit, distance, float('inf')))
        if polygons is not None and self.cfg.scene_spec is not None:
            from .geometry import polygon_raycast
            ranges = torch.minimum(ranges,polygon_raycast(origin,directions,polygons,polygon_counts))
        return ranges

    def observe(self, xy, yaw, obstacles, valid, polygons=None, polygon_counts=None, rasterize=True):
        due = (self.sample_tick < 0) | (self.ticks - self.sample_tick >= self.period)
        ids = due.nonzero().flatten()
        if ids.numel():
            cfg = self.cfg
            ranges = self.raycast(xy[ids], yaw[ids], obstacles[ids], valid[ids],
                                  None if polygons is None else polygons[ids],
                                  None if polygon_counts is None else polygon_counts[ids])
            # Out-of-range hits remain invalid rather than being clamped into
            # fictional maximum-range returns that would clear unknown space.
            measured = (ranges >= cfg.lidar_range_min) & (ranges <= cfg.lidar_range_max)
            if cfg.lidar_noise_std or cfg.lidar_noise_std_range is not None:
                ranges += torch.randn(ranges.shape, device=self.device, dtype=ranges.dtype,
                                      generator=self.generator) * self.noise_std[ids]
            if cfg.lidar_outlier_probability:
                outlier = torch.rand(ranges.shape, device=self.device, generator=self.generator) < cfg.lidar_outlier_probability
                error = (2*torch.rand(ranges.shape, device=self.device, generator=self.generator)-1)*cfg.lidar_outlier_error
                ranges += outlier*error
            if cfg.lidar_dropout or cfg.lidar_dropout_range is not None:
                measured &= torch.rand(ranges.shape, device=self.device, generator=self.generator) >= self.dropout[ids]
            measured &= ~self.sector_mask[ids]
            measured &= (ranges >= cfg.lidar_range_min) & (ranges <= cfg.lidar_range_max)
            ranges = torch.where(measured, ranges, float('inf'))
            self.ranges[ids] = ranges
            self.sample_tick[ids] = self.ticks[ids]
        if not rasterize:
            return self.ranges
        grid_ids=(self.grid_tick != self.sample_tick).nonzero().flatten()
        if grid_ids.numel():
            # Rasterization is optional for sector policies; the cached grid is
            # rebuilt only for explicit visualization/legacy observations.
            ranges=self.ranges[grid_ids]
            normalized = torch.minimum(ranges[:, self.lower], ranges[:, self.upper])
            points = normalized[..., None] * self.normalized_directions + self.origin
            points = torch.where(torch.isfinite(normalized)[..., None], points, float('nan'))
            self.grid[grid_ids] = scan_grid(points, self.origin)
            self.grid_tick[grid_ids] = self.sample_tick[grid_ids]
        return self.grid.clone()
