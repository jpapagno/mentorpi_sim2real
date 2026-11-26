"""Observation and action adapter for the Isaac Lab MentorPi Motion task."""

import math

import numpy as np

ARCHITECTURE = "isaaclab_motion_ppo_v1"
RAY_COUNT = 72
RANGE_M = 3.0
ROOT_HEIGHT_M = 0.04
# In the training ray caster the hit plane is 6 cm above the reported scan origin.
RAY_HEIGHT_DELTA_M = 0.06
PERIOD_S = 1 / 60
CONTRACT = {
    "architecture": ARCHITECTURE,
    "observation_contract": "world_pose3_velocity3_goal2_lidar72_v1",
    "policy_shape": [1, 8 + RAY_COUNT],
    "action_shape": [1, 2],
    "speed_scale_mps": 1.0,
    "dt": PERIOD_S,
}


BOUNDED_CONTRACT = {**CONTRACT, "action_limit_mps": 0.6, "ray_height_delta_m": 0.01}


def rotation(yaw):
    """Rotate body-frame XY into odometry XY."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def observation(base, start, body_velocity, goal, body_scan_points, ray_height_delta=RAY_HEIGHT_DELTA_M):
    """Use the starting pose as the simulated world frame on the robot."""
    points = np.asarray(body_scan_points)[:, :2]
    if not len(points):
        raise RuntimeError("No valid LiDAR returns")
    body_to_world = rotation(base[2] - start[2])
    world_points = points @ body_to_world.T
    angles = np.arctan2(world_points[:, 1], world_points[:, 0])
    bins = np.floor((angles + np.pi) * RAY_COUNT / (2 * np.pi) + 0.5).astype(int) % RAY_COUNT
    body_ranges = np.linalg.norm(points, axis=1)
    ranges = np.full(RAY_COUNT, RANGE_M)
    np.minimum.at(ranges, bins, np.hypot(body_ranges, ray_height_delta))
    odom_to_world = rotation(start[2]).T
    position = odom_to_world @ (base[:2] - start[:2])
    velocity = body_to_world @ body_velocity
    offset = odom_to_world @ (goal - base[:2])
    state = np.asarray([*position, ROOT_HEIGHT_M, *velocity, 0.0, *offset, *ranges], dtype=np.float32)
    return state[None, :]


def body_command(action, relative_yaw):
    """Convert clipped world XY m/s to body XY m/s with zero commanded yaw."""
    xy = rotation(relative_yaw).T @ (action * CONTRACT["speed_scale_mps"])
    return np.array([*xy, 0.0])


def goal_reached(odom_offset, start_yaw):
    return bool(np.all(np.abs(rotation(start_yaw).T @ odom_offset) <= 0.15))
