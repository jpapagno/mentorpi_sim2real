"""Evaluate fixed numbers of episodes per seed and persist reproducible results."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time

import torch

from mentorpi_nav.core import NavigationCfg, NavigationState
from mentorpi_nav.evaluation import evaluate_actor
from mentorpi_nav.playback import MODEL_ID, load_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[456, 789, 2026])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", action="store_true", help="Copy the source checkpoint before evaluation")
    parser.add_argument("--scenario-file", default=None, help="Explicit challenge scenario override")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source.with_name("task.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL_ID:
        raise ValueError("Unsupported policy contract")
    checkpoint = source
    if args.snapshot:
        checkpoint = output / "baseline.pt"
        shutil.copy2(source, checkpoint)
        (output / "task.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg = NavigationCfg(**manifest["navigation"])
    if args.scenario_file:
        cfg = replace(cfg, **json.loads(Path(args.scenario_file).read_text(encoding="utf-8")))
    dummy = NavigationState(1, cfg, device=args.device)
    actor = load_actor(checkpoint, dummy.observations(), args.device)
    summary = {"source_checkpoint": str(source), "evaluated_checkpoint": str(checkpoint),
               "episodes_per_seed": args.episodes, "navigation": asdict(cfg), "results": []}
    for seed in args.seeds:
        started = time.monotonic()
        result = evaluate_actor(actor, cfg, seed, args.episodes, args.device,
                                cfg.episode_seconds)
        (output / f"seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        compact = {k: v for k, v in result.items() if k not in {"episode_results", "navigation"}}
        compact["evaluation_wall_seconds"] = time.monotonic() - started
        summary["results"].append(compact)
        summary["meets_target"] = all(r["success_rate"] > .90 and r["collision_rate"] <= .02 for r in summary["results"])
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(compact), flush=True)


if __name__ == "__main__":
    main()
