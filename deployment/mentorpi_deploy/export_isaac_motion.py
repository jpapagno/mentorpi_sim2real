"""Export an Isaac Lab MentorPi Motion checkpoint for the Pi runtime."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from . import isaac_motion


def export_checkpoint(checkpoint, output):
    run = checkpoint.parent
    agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    metadata = json.loads((run / "run.json").read_text())
    actor_cfg = agent["actor"]
    lidar = env["scene"]["lidar"]
    distribution = actor_cfg["distribution_cfg"]
    distribution_name = distribution["class_name"]
    action = env["actions"]["planar_velocity"]
    ray_offset = list(map(float, lidar["offset"]["pos"]))
    distribution_kwargs = {"class_name": distribution_name, "init_std": float(distribution["init_std"])}
    if distribution_name == "GaussianDistribution":
        contract_fields = isaac_motion.CONTRACT
        action_matches = (
            float(action.get("speed_scale", "nan")) == contract_fields["speed_scale_mps"]
            and ray_offset == [0.0, 0.0, isaac_motion.RAY_HEIGHT_DELTA_M]
        )
        distribution_kwargs["std_type"] = distribution["std_type"]
    elif distribution_name == "SquashedGaussianDistribution":
        contract_fields = isaac_motion.BOUNDED_CONTRACT
        limit = contract_fields["action_limit_mps"]
        action_range = tuple(map(float, distribution["action_range"]))
        action_matches = (
            action_range == (-limit, limit)
            and float(action.get("speed_limit", "nan")) == limit
            and ray_offset == [0.0, 0.0, contract_fields["ray_height_delta_m"]]
        )
        distribution_kwargs["action_range"] = action_range
        distribution_kwargs["std_range"] = tuple(map(float, distribution["std_range"]))
    else:
        raise ValueError(f"Unsupported actor distribution: {distribution_name}")
    observation_names = [name for name, value in env["observations"]["policy"].items() if isinstance(value, dict)]
    if (
        metadata["task"] != "MentorpiIsaaclab-Balance-Mentorpi"
        or agent["obs_groups"]["actor"] != ["policy"]
        or actor_cfg["class_name"] != "MLPModel"
        or actor_cfg["obs_normalization"].lower() != "false"
        or not action_matches
        or observation_names != ["position", "linear_velocity", "goal_offset", "lidar"]
        or float(env["sim"]["dt"]) * int(env["decimation"]) != isaac_motion.PERIOD_S
        or lidar["ray_alignment"] != "yaw"
        or int(lidar["pattern_cfg"]["channels"]) != 1
        or list(map(float, lidar["pattern_cfg"]["horizontal_fov_range"])) != [-180.0, 180.0]
        or float(lidar["pattern_cfg"]["horizontal_res"]) != 360 / isaac_motion.RAY_COUNT
        or float(lidar["max_distance"]) != isaac_motion.RANGE_M
    ):
        raise ValueError("Run settings do not match the Isaac Lab MentorPi Motion deployment contract")

    size = isaac_motion.CONTRACT["policy_shape"][1]
    samples = torch.rand((64, size), generator=torch.Generator().manual_seed(42))
    samples[:, :8] = 2 * samples[:, :8] - 1
    samples[:, 2] = isaac_motion.ROOT_HEIGHT_M
    samples[:, 5] = 0.0
    samples[:, 8:] *= isaac_motion.RANGE_M
    samples[::2, 8::3] = isaac_motion.RANGE_M
    obs = TensorDict({"policy": samples}, batch_size=[len(samples)])
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        output_dim=2,
        hidden_dims=list(map(int, actor_cfg["hidden_dims"])),
        activation=actor_cfg["activation"],
        distribution_cfg=distribution_kwargs,
    ).eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    scripted = torch.jit.script(actor.as_jit().eval())

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        model_file = bundle / "policy.ts"
        scripted.save(str(model_file))
        reloaded = torch.jit.load(str(model_file), map_location="cpu").eval()
        with torch.inference_mode():
            expected = actor(obs)
            actual = reloaded(samples)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        np.savez_compressed(bundle / "parity_cases.npz", policy=samples.numpy(), action=expected.numpy())
        contract = {
            **contract_fields,
            "source_checkpoint": f"{run.name}/{checkpoint.name}",
            "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_task": metadata["task"],
            "torch_export_max_error": float((actual - expected).abs().max()),
        }
        (bundle / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        bundle.rename(output)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Training checkpoint beside params/ and run.json"
    )
    parser.add_argument("--output", type=Path, required=True, help="New policy bundle directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    torch.set_num_threads(1)
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
