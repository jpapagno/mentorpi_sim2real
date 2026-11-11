"""Training-only demonstration/DAgger warm start for the existing CNN actor.

Saved policies run by themselves in mentorpi_nav.cli.evaluate and the viewer. Planner
actions are used only to generate supervised training labels in this script.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from mentorpi_nav.core import NavigationCfg, NavigationState
from mentorpi_nav.playback import load_actor
from mentorpi_nav.model_spec import CNN_CONFIG, HIDDEN_DIMS
from mentorpi_nav.teacher import PathTeacher


def collect(cfg, layouts, seed, actor=None):
    grids, states, targets = [], [], []
    outcomes = {"successes": 0, "collisions": 0, "timeouts": 0, "missing_paths": 0}
    started = time.monotonic()
    for batch in range(math.ceil(layouts/32)):
        size = min(32, layouts-batch*32)
        state = NavigationState(size, cfg, seed=seed+batch)
        state.reset()
        teacher = PathTeacher(state)
        outcomes["missing_paths"] += sum(p is None for p in teacher.paths)
        active = torch.ones(size, dtype=torch.bool)
        for step in range(math.ceil(cfg.episode_seconds/cfg.dt)):
            if not active.any():
                break
            if actor is not None and step and step % 40 == 0:
                teacher.replan(state, active)
            label = teacher.actions(state, active)
            observations = None
            if step % 4 == 0 or actor is not None:
                observations = state.observations()
            if step % 4 == 0:
                pixels = observations["occupancy"][active].numpy().reshape(-1,10000).astype(np.uint8)
                grids.append(np.packbits(pixels, axis=-1))
                states.append(observations["policy"][active].numpy().copy())
                targets.append(label[active].numpy().copy())
            if actor is None:
                action = label
            else:
                device = next(actor.parameters()).device
                with torch.no_grad():
                    inputs = TensorDict({k:v.to(device) for k,v in observations.items()},batch_size=[size])
                    action = actor(inputs).clamp(-1,1).cpu()
            action[~active] = 0
            state.velocity[~active] = 0
            state.step(action)
            outcomes["successes"] += int((state.success & active).sum())
            outcomes["collisions"] += int((state.collision & active).sum())
            active &= ~(state.success | state.collision)
        outcomes["timeouts"] += int(active.sum())
        print(json.dumps(dict(layouts_complete=min((batch+1)*32,layouts),elapsed_s=time.monotonic()-started,
                              **outcomes)),flush=True)
    return {"grids": np.concatenate(grids), "states": np.concatenate(states),
            "targets": np.concatenate(targets)}, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--layouts", type=int, default=512)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dagger", action="store_true", help="Collect learner-state labels and merge prior demonstrations")
    parser.add_argument("--prior-data", default=None)
    parser.add_argument("--reuse-data", default=None, help="Reuse a previously collected packed dataset")
    parser.add_argument("--goal-residual", action="store_true", help="Train a fresh goal-conditioned residual CNN")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source.with_name("task.json").read_text())
    cfg = NavigationCfg(**manifest["navigation"])
    if args.scenario_file:
        cfg = replace(cfg, **json.loads(Path(args.scenario_file).read_text()))
    dummy = NavigationState(1,cfg,device=args.device)
    actor = load_actor(source,dummy.observations(),args.device)
    if args.reuse_data:
        with np.load(args.reuse_data) as saved_data:
            data={k:saved_data[k].copy() for k in ("grids","states","targets")}
        outcomes={"reused_data":args.reuse_data}
    else:
        print("Collecting training-only demonstrations...",flush=True)
        data, outcomes = collect(cfg,args.layouts,args.seed,actor if args.dagger else None)
    if args.prior_data:
        with np.load(args.prior_data) as prior:
            data = {k:np.concatenate((prior[k],v)) for k,v in data.items()}
    # A missing teacher path is an unavailable label, not a command to stop.
    labelled = np.linalg.norm(data["targets"][:,:2], axis=-1) > .05
    data = {k:v[labelled] for k,v in data.items()}
    np.savez_compressed(output/"demonstrations.npz",**data)
    print(json.dumps(dict(samples=len(data["states"]),collection_outcomes=outcomes)),flush=True)
    manifest.update(navigation=asdict(cfg),training_method="dagger" if args.dagger else "behavior_cloning",
                    source_checkpoint=str(source),episode_seconds=cfg.episode_seconds,
                    training_seed=args.seed,demonstration_samples=len(data["states"]),
                    teacher_clearance_m=.08, teacher_fallback_clearance_m=.02)
    if args.goal_residual:
        from mentorpi_nav.networks import GoalResidualCNN
        actor=GoalResidualCNN(TensorDict(dummy.observations(),batch_size=[1]),
            {"actor":["policy","occupancy"]},"actor",3,hidden_dims=HIDDEN_DIMS,activation="elu",
            obs_normalization=False,cnn_cfg=CNN_CONFIG,
            distribution_cfg={"class_name":"GaussianDistribution","init_std":.15}).to(args.device)
        manifest["network_architecture"]="goal_residual_cnn_v1"
    (output/"task.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    source_weights = torch.load(source,map_location="cpu",weights_only=True)
    if manifest.get("network_architecture") == "goal_residual_cnn_v1":
        from mentorpi_nav.networks import GoalResidualCNN
        critic=GoalResidualCNN(TensorDict(dummy.observations(),batch_size=[1]),
            {"critic":["policy","occupancy"]},"critic",1,hidden_dims=HIDDEN_DIMS,activation="elu",
            obs_normalization=False,cnn_cfg=CNN_CONFIG)
        source_weights["critic_state_dict"]=critic.state_dict()
    optimizer = torch.optim.Adam(actor.parameters(),lr=1e-4)
    actor.train()
    history=[]
    started=time.monotonic()
    for step in range(1,args.updates+1):
        index=rng.integers(0,len(data["states"]),size=256)
        pixels=np.unpackbits(data["grids"][index],axis=-1,count=10000).reshape(-1,1,100,100)
        state_values=data["states"][index].copy()
        target_values=data["targets"][index].copy()
        # Exact symmetries of the disc robot / equal x-y speed limits.
        # Rotate map, goal, motion history and desired translation together.
        turns=int(rng.integers(0,4))
        pixels=np.rot90(pixels,turns,axes=(-2,-1)).copy()
        for _ in range(turns):
            for x,y in ((0,1),(2,3),(5,6)):
                state_values[:,[x,y]]=np.stack((-state_values[:,y],state_values[:,x]),axis=-1)
            target_values[:,:2]=np.stack((-target_values[:,1],target_values[:,0]),axis=-1)
        if rng.random()<.5:
            pixels=pixels[...,::-1].copy()
            state_values[:,[1,3,4,6,7]] *= -1
            target_values[:,[1,2]] *= -1
        # The teacher's desired command is pose/route-based. Decorrelate history
        # from the label so the student cannot succeed by copying the previous
        # expert command while ignoring the goal and map.
        history_choice=rng.random(len(index))
        state_values[history_choice<1/3,2:]=0
        random_mask=history_choice>2/3
        state_values[random_mask,2:]=rng.uniform(-1,1,(int(random_mask.sum()),6))
        obs=TensorDict({"policy":torch.from_numpy(state_values).to(args.device),
                       "occupancy":torch.from_numpy(pixels).to(args.device,dtype=torch.float32)},batch_size=[len(index)])
        target=torch.from_numpy(target_values).to(args.device)
        prediction=actor(obs)
        loss=(prediction-target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0)
        optimizer.step()
        if step % 200 == 0:
            item=dict(update=step,mean_squared_error=float(loss.detach()),elapsed_s=time.monotonic()-started)
            history.append(item)
            print(json.dumps(item),flush=True)
            (output/"training_history.json").write_text(json.dumps(history,indent=2))
        if step % 1000 == 0 or step == args.updates:
            # Critic is retained only as an initialization for optional PPO
            # transfer. No obsolete optimizer moments are copied into this model.
            saved={"actor_state_dict":{k:v.detach().cpu() for k,v in actor.state_dict().items()},
                   "critic_state_dict":source_weights["critic_state_dict"],"iter":0,"infos":manifest["training_method"]}
            torch.save(saved,output/f"model_{step}.pt")
            torch.save(saved,output/"final.pt")
    print(f"Learned actor checkpoint: {output/'final.pt'}",flush=True)


if __name__ == "__main__":
    main()
