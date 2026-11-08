"""Run, train, or evaluate MentorPi navigation inside the installed Isaac Lab 3.

Use the Isaac Lab venv interpreter, not the system Python.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from datetime import datetime

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict


def main():
    paths_file = Path(__file__).with_name("local_paths.json")
    local_paths = json.loads(paths_file.read_text(encoding="utf-8")) if paths_file.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train", "evaluate"], default="smoke")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--scenario-file", default=None, help="JSON overrides for NavigationCfg")
    parser.add_argument("--transfer-scenario", action="store_true", help="Transfer model weights to a changed scenario; reset optimizer and iteration")
    parser.add_argument("--map-resolution", type=float, default=None, help="Metres per grid cell (100 cells per side)")
    parser.add_argument("--map-inflation", type=float, default=None, help="Optional binary obstacle/wall inflation in metres")
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--discount", type=float, default=.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fixed-learning-rate", action="store_true", help="Disable adaptive learning-rate increases during fine-tuning")
    parser.add_argument("--entropy-coef", type=float, default=.005)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25, help="Save a policy checkpoint every X PPO iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", default=local_paths.get("mentorpi_asset"), help="USD visual asset; optional for training")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log-dir", default=None)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("--mode evaluate requires --checkpoint")
    if min(args.steps, args.iterations, args.num_envs, args.save_every) < 1:
        parser.error("steps, iterations, num-envs, and save-every must be positive")

    import torch
    from mentorpi_nav.lab_env import MentorPiEnv, MentorPiEnvCfg
    from mentorpi_nav.agent import MentorPiPPOCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = MentorPiEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device or "cuda:0"
    cfg.scene.num_envs = args.num_envs
    if args.scenario_file:
        cfg.navigation = replace(cfg.navigation, **json.loads(Path(args.scenario_file).read_text(encoding="utf-8")))
    for name, value in [("obstacle_count", args.obstacles), ("occupancy_resolution", args.map_resolution),
                        ("occupancy_inflation", args.map_inflation)]:
        if value is not None:
            cfg.navigation = replace(cfg.navigation, **{name: value})
    cfg.asset_path = args.asset
    source_manifest = None
    if args.checkpoint:
        source_manifest = json.loads(Path(args.checkpoint).resolve().with_name("task.json").read_text(encoding="utf-8"))
        if args.mode == "train" and source_manifest.get("training_method") in {"behavior_cloning", "dagger"} and not args.transfer_scenario:
            parser.error("Supervised checkpoints require --transfer-scenario for PPO with a fresh optimizer")
    agent_cfg = MentorPiPPOCfg()
    if source_manifest and source_manifest.get("network_architecture") == "goal_residual_cnn_v1":
        agent_cfg.actor.class_name = "mentorpi_nav.networks:GoalResidualCNN"
        agent_cfg.critic.class_name = "mentorpi_nav.networks:GoalResidualCNN"
    agent_cfg.device = cfg.sim.device
    agent_cfg.seed = args.seed
    agent_cfg.save_interval = args.save_every
    agent_cfg.num_steps_per_env = args.rollout_steps
    agent_cfg.algorithm.gamma = args.discount
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.algorithm.entropy_coef = args.entropy_coef
    if args.fixed_learning_rate:
        agent_cfg.algorithm.schedule = "fixed"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or f"logs/{args.mode}_{stamp}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    with launch_simulation(cfg, args):
        env = MentorPiEnv(cfg)
        try:
            manifest = env.task_manifest()
            manifest.update(seed=args.seed, num_envs=args.num_envs, save_every=args.save_every,
                            episode_seconds=cfg.navigation.episode_seconds,
                            source_checkpoint=args.checkpoint, transfer_scenario=args.transfer_scenario)
            if source_manifest and "network_architecture" in source_manifest:
                manifest["network_architecture"] = source_manifest["network_architecture"]
            if args.checkpoint:
                _validate_checkpoint(args.checkpoint, manifest, args.transfer_scenario)
            manifest_path = log_dir / "task.json"
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
                if old["model"] != manifest["model"] or old["navigation"] != json.loads(json.dumps(manifest["navigation"])):
                    raise ValueError("Output directory contains a different task; choose a new --log-dir")
            (log_dir / "task.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (log_dir / "agent.json").write_text(json.dumps(to_dict(agent_cfg), indent=2, default=str), encoding="utf-8")
            if args.mode == "train":
                wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
                runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), str(log_dir), device=cfg.sim.device)
                if args.checkpoint:
                    load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False} if args.transfer_scenario else None
                    runner.load(args.checkpoint, load_cfg=load_cfg)
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
                runner.save(str(log_dir / "final.pt"))
                print(f"Checkpoint: {log_dir / 'final.pt'}")
            else:
                obs, _ = env.reset()
                from mentorpi_nav.preview import save_grid_preview
                policy = None
                if args.checkpoint:
                    from mentorpi_nav.playback import load_actor
                    policy = load_actor(args.checkpoint, obs, cfg.sim.device)
                save_grid_preview(env.nav, log_dir / "occupancy.png")
                successes = collisions = timeouts = episodes = 0
                executed_steps = 0
                for _ in range(args.steps):
                    if not env.sim.is_running():
                        break
                    with torch.inference_mode():
                        if policy:
                            from tensordict import TensorDict
                            action = policy(TensorDict(obs, batch_size=[args.num_envs]))
                        else:
                            action = 2 * torch.rand(args.num_envs, 3, device=env.device) - 1
                        obs, reward, terminated, truncated, extras = env.step(action)
                        executed_steps += 1
                        assert obs["policy"].shape == (args.num_envs, cfg.navigation.observation_size)
                        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                        assert obs["occupancy"].shape == (args.num_envs, 1, 100, 100)
                        assert ((obs["occupancy"] == 0) | (obs["occupancy"] == 1)).all()
                        successes += int(env.outcomes["success"].sum())
                        collisions += int(env.outcomes["collision"].sum())
                        timeouts += int(truncated.sum())
                        episodes += int((terminated | truncated).sum())
                result = dict(mode=args.mode, seed=args.seed, steps_executed=executed_steps, completed_episodes=episodes,
                              successes=successes, collisions=collisions, timeouts=timeouts,
                              success_rate=successes / episodes if episodes else None)
                (log_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps(result, indent=2))
        finally:
            env.close()


def _validate_checkpoint(checkpoint, current, transfer=False):
    """Refuse incompatible semantics even if tensors have the same shape."""
    previous = json.loads(Path(checkpoint).resolve().with_name("task.json").read_text(encoding="utf-8"))
    from dataclasses import asdict
    from mentorpi_nav.core import NavigationCfg
    old = json.loads(json.dumps(asdict(NavigationCfg(**previous["navigation"]))))
    new = json.loads(json.dumps(asdict(NavigationCfg(**current["navigation"]))))
    invariant = ["robot_radius", "arena_half_extent", "speed_limits", "acceleration_limits", "dt",
                 "occupancy_size", "occupancy_resolution", "occupancy_inflation", "goal_tolerance"]
    differs = any(old[k] != new[k] for k in invariant) if transfer else old != new
    if previous["model"] != current["model"] or differs:
        raise ValueError("Checkpoint task configuration differs; use the same navigation settings")


if __name__ == "__main__":
    main()
