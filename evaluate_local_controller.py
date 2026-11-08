"""Evaluate learned velocity control with global-path subgoals, matching deployment.

The A* route serves the role of Nav2's global planner. It supplies subgoals only;
every velocity command is produced by the actor, without teacher actions or a
collision override. This is a stack test, separate from direct-goal benchmarks.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from tensordict import TensorDict
from mentorpi_nav.core import NavigationCfg,NavigationState,world_to_body
from mentorpi_nav.playback import load_actor
from mentorpi_nav.teacher import plan_path
from mentorpi_nav.evaluation import wilson_interval
from deployment.runtime.navigation_math import subgoal_on_path


def dense_path(path):
    points=[path[0]]
    for start,end in zip(path,path[1:]):
        start,end=np.asarray(start),np.asarray(end)
        count=max(1,math.ceil(float(np.linalg.norm(end-start))/.05))
        points.extend((start+(end-start)*(index/count)).tolist() for index in range(1,count+1))
    return np.asarray(points)


def evaluate(actor,cfg,seed,episodes,device,lookahead):
    state=NavigationState(episodes,cfg,device,seed);state.reset()
    paths=[]
    for start,goal,obstacles,valid in zip(state.xy.cpu().tolist(),state.goal.cpu().tolist(),state.obstacles.cpu().tolist(),state.valid.cpu().tolist()):
        route=plan_path(start,goal,[o for o,v in zip(obstacles,valid) if v],cfg)
        paths.append(dense_path(route) if route else None)
    active=torch.tensor([p is not None for p in paths],device=device,dtype=torch.bool)
    success=torch.zeros_like(active);collision=torch.zeros_like(active)
    max_steps=math.ceil(cfg.episode_seconds/cfg.dt)
    steps=torch.zeros(episodes,device=device,dtype=torch.int64)
    with torch.no_grad():
        for step in range(max_steps):
            if not active.any():break
            positions=state.xy.cpu().numpy()
            enabled=active.cpu().numpy()
            targets=np.asarray([subgoal_on_path(path,position,lookahead) if valid else position
                                for path,position,valid in zip(paths,positions,enabled)],dtype=np.float32)
            obs=state.observations()
            obs['policy'][:,:2]=world_to_body(torch.from_numpy(targets).to(device)-state.xy,state.yaw)/(4*math.sqrt(2))
            action=actor(TensorDict(obs,batch_size=[episodes])).clamp(-1,1)
            action[~active]=0;state.velocity[~active]=0
            state.step(action);steps[active]=step+1
            success|=state.success&active;collision|=state.collision&active
            active&=~(state.success|state.collision)
    successes=int(success.sum());collisions=int(collision.sum())
    return {'seed':seed,'episodes':episodes,'successes':successes,'collisions':collisions,
        'timeouts':int(active.sum()),'unplanned':sum(p is None for p in paths),'success_rate':successes/episodes,
        'collision_rate':collisions/episodes,'success_95pct_wilson':wilson_interval(successes,episodes),
        'mean_episode_seconds':float(steps.float().mean())*cfg.dt}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--scenario-file',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[456,789,2026])
    parser.add_argument('--episodes',type=int,default=200);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--lookahead',type=float,default=1.);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    manifest=json.loads(Path(args.checkpoint).with_name('task.json').read_text())
    cfg=replace(NavigationCfg(**manifest['navigation']),**json.loads(Path(args.scenario_file).read_text()))
    actor=load_actor(args.checkpoint,NavigationState(1,cfg,args.device).observations(),args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    summary={'checkpoint':args.checkpoint,'scenario_file':args.scenario_file,'lookahead_m':args.lookahead,
        'mode':'global_path_subgoal_stack','velocity_source':'learned_actor_only','results':[]}
    for seed in args.seeds:
        started=time.monotonic()
        result=evaluate(actor,cfg,seed,args.episodes,args.device,args.lookahead)
        result['wall_seconds']=time.monotonic()-started
        summary['results'].append(result)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
