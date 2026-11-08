"""Measure the training-only planner/controller on the same complete-episode task."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from mentorpi_nav.core import NavigationCfg, NavigationState
from mentorpi_nav.teacher import PathTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = NavigationCfg()
    if args.scenario_file:
        cfg = replace(cfg, **json.loads(Path(args.scenario_file).read_text()))
    state = NavigationState(args.episodes, cfg, seed=314159)
    state.reset()
    teacher = PathTeacher(state)
    active = torch.ones(args.episodes, dtype=torch.bool)
    successes = torch.zeros_like(active)
    collisions = torch.zeros_like(active)
    with torch.no_grad():
        for _ in range(round(cfg.episode_seconds/cfg.dt)):
            if not active.any():
                break
            actions = teacher.actions(state)
            actions[~active] = 0
            state.velocity[~active] = 0
            state.step(actions)
            successes |= state.success & active
            collisions |= state.collision & active
            active &= ~(state.success | state.collision)
    print(json.dumps(dict(episodes=args.episodes,successes=int(successes.sum()),collisions=int(collisions.sum()),
                          timeouts=int(active.sum()),missing_paths=sum(p is None for p in teacher.paths))))


if __name__ == "__main__":
    main()
