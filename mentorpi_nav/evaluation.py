"""Fixed-episode batched evaluation of the exact task dynamics, without rendering.

Every sampled episode contributes exactly one outcome. No fast-success bias
from counting only episodes that happen to finish within a wall-clock rollout.
"""

from dataclasses import asdict
import math

import torch
from tensordict import TensorDict

from .core import NavigationState


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    centre = (p + z*z / (2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [centre - radius, centre + radius]


def evaluate_actor(actor, cfg, seed, episodes=200, device="cpu", episode_seconds=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    state = NavigationState(episodes, cfg, device, seed)
    state.reset()
    initial_xy = state.xy.clone()
    initial_goal = state.goal.clone()
    initial_yaw = state.yaw.clone()
    initial_obstacles = state.obstacles.clone()
    initial_valid = state.valid.clone()
    initial_distance = torch.linalg.vector_norm(state.goal - state.xy, dim=-1)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    success = torch.zeros_like(active)
    collision = torch.zeros_like(active)
    steps = torch.zeros(episodes, dtype=torch.long, device=device)
    path_length = torch.zeros(episodes, device=device)
    final_distance = initial_distance.clone()
    episode_seconds = cfg.episode_seconds if episode_seconds is None else episode_seconds
    max_steps = math.ceil(episode_seconds / cfg.dt)
    with torch.no_grad():
        for step in range(max_steps):
            if not active.any():
                break
            observations = TensorDict(state.observations(), batch_size=[episodes])
            actions = actor(observations).clamp(-1, 1)
            actions[~active] = 0
            state.velocity[~active] = 0
            before = state.xy.clone()
            state.step(actions)
            path_length += torch.linalg.vector_norm(state.xy - before, dim=-1) * active
            steps[active] = step + 1
            final_distance[active] = torch.linalg.vector_norm(state.goal - state.xy, dim=-1)[active]
            success |= state.success & active
            collision |= state.collision & active
            active &= ~(state.success | state.collision)
    timeout = ~(success | collision)
    rows = []
    arrays = {name: tensor.cpu().tolist() for name, tensor in {
        "success": success, "collision": collision, "timeout": timeout,
        "steps": steps, "initial_distance": initial_distance, "final_distance": final_distance,
        "path_length": path_length, "start": initial_xy, "goal": initial_goal, "yaw": initial_yaw,
        "obstacles": initial_obstacles, "valid": initial_valid,
    }.items()}
    for i in range(episodes):
        rows.append({name: values[i] for name, values in arrays.items()})
    successes = int(success.sum())
    collisions = int(collision.sum())
    return {
        "seed": seed, "episodes": episodes, "successes": successes, "collisions": collisions,
        "timeouts": int(timeout.sum()), "success_rate": successes / episodes,
        "collision_rate": collisions / episodes, "timeout_rate": float(timeout.float().mean()),
        "success_95pct_wilson": wilson_interval(successes, episodes),
        "mean_episode_seconds": float(steps.float().mean()) * cfg.dt,
        "mean_final_distance": float(final_distance.mean()),
        "navigation": asdict(cfg), "episode_seconds": episode_seconds, "episode_results": rows,
    }
