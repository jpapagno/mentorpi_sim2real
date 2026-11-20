"""Smoke-test or train LiDAR navigation without Isaac, USD or a renderer."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

import torch

from mentorpi_nav.core import NavigationCfg
from mentorpi_nav.planar_env import RslPlanarEnv, task_manifest
from mentorpi_nav.model_spec import planar_ppo_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','train'],default='smoke')
    parser.add_argument('--scenario-file',default=None)
    parser.add_argument('--checkpoint')
    parser.add_argument('--network',choices=['goal_residual_cnn_v1','cnn_v1'],default=None,
                        help='Fresh policy architecture; cnn_v1 has no built-in goal-seeking prior')
    parser.add_argument('--transfer-scenario',action='store_true')
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--iterations',type=int,default=500)
    parser.add_argument('--rollout-steps',type=int,default=32)
    parser.add_argument('--save-every',type=int,default=25)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--gamma',type=float,default=.99,help='Future reward discount per control step')
    parser.add_argument('--entropy-coef',type=float,default=.001,help='PPO exploration entropy bonus coefficient')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 0 < args.gamma < 1:
        parser.error('--gamma must be between zero and one')
    if not math.isfinite(args.entropy_coef) or args.entropy_coef<0:
        parser.error('--entropy-coef must be finite and nonnegative')
    if min(args.num_envs,args.steps,args.iterations,args.rollout_steps,args.save_every)<1:
        parser.error('Counts must be positive')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    source=json.loads(Path(args.checkpoint).with_name('task.json').read_text()) if args.checkpoint else None
    cfg=NavigationCfg(**source['navigation']) if source else NavigationCfg()
    if args.scenario_file:
        cfg=replace(cfg,**json.loads(Path(args.scenario_file).read_text()))
    if args.mode=='train' and cfg.policy_input!='occupancy':
        parser.error('Laser-sector training uses python -m mentorpi_nav.cli.td3')
    if args.checkpoint and not args.transfer_scenario and cfg != NavigationCfg(**source['navigation']):
        parser.error('Changed scenario requires --transfer-scenario')
    if source and source['model']!='analytic_swept_disc_occupancy_v2':
        parser.error('Unsupported checkpoint observation contract')
    if source and args.transfer_scenario:
        old=asdict(NavigationCfg(**source['navigation']));new=asdict(cfg)
        invariant=['policy_input','robot_radius','robot_size','goal_normalization_half_extent','arena_half_extent','speed_limits','acceleration_limits','dt',
                   'occupancy_size','occupancy_resolution','occupancy_inflation','goal_tolerance']
        if any(old[k]!=new[k] for k in invariant):
            parser.error('Transfer must preserve policy shape, normalization, action scale and timing')
    architecture=source.get('network_architecture','cnn_v1') if source else (args.network or 'goal_residual_cnn_v1')
    if source and args.network and args.network!=architecture:
        parser.error('--network must match the checkpoint architecture')
    residual=architecture=='goal_residual_cnn_v1'
    if residual and ((cfg.goal_normalization_half_extent or cfg.arena_half_extent)!=2 or tuple(cfg.speed_limits)!=(.2,.2,.5)):
        parser.error('Residual v1 requires original arena normalization and speed limits')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=task_manifest(cfg)
    manifest.update(seed=args.seed,num_envs=args.num_envs,source_checkpoint=args.checkpoint,
                    transfer_scenario=args.transfer_scenario,training_method='ppo',
                    network_architecture='goal_residual_cnn_v1' if residual else 'cnn_v1')
    if (output/'task.json').exists():
        parser.error('Choose a fresh output directory to preserve prior runs')
    (output/'task.json').write_text(json.dumps(manifest,indent=2))
    agent=planar_ppo_config(args.seed,args.device,args.rollout_steps,args.save_every,args.learning_rate,residual,gamma=args.gamma,entropy_coef=args.entropy_coef)
    (output/'agent.json').write_text(json.dumps(agent,indent=2))
    env=RslPlanarEnv(cfg,args.num_envs,args.device,args.seed)
    try:
        if args.mode=='train':
            from rsl_rl.runners import OnPolicyRunner
            runner=OnPolicyRunner(env,agent,str(output),device=args.device)
            if args.checkpoint:
                runner.load(args.checkpoint,load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False} if args.transfer_scenario else None)
            runner.learn(args.iterations,init_at_random_ep_len=False)
            runner.save(str(output/'final.pt'))
        else:
            totals={'successes':0,'collisions':0,'timeouts':0}
            actor=None
            if args.checkpoint:
                from mentorpi_nav.playback import load_actor
                actor=load_actor(args.checkpoint,env.batch.nav.observations(),args.device)
            with torch.no_grad():
                for _ in range(args.steps):
                    action=actor(env.get_observations()) if actor else 2*torch.rand(args.num_envs,3,device=args.device)-1
                    _,_,_,info=env.step(action)
                    totals['successes']+=int(info['outcomes']['success'].sum())
                    totals['collisions']+=int(info['outcomes']['collision'].sum())
                    totals['timeouts']+=int(info['time_outs'].sum())
            from mentorpi_nav.preview import save_scene_preview
            save_scene_preview(env.batch.nav,output/'scene.png')
            (output/'result.json').write_text(json.dumps({'steps':args.steps,**totals},indent=2))
        forbidden=[m for m in sys.modules if m.split('.')[0] in {'isaaclab','isaacsim','pxr','newton','omni'}]
        if forbidden:
            raise RuntimeError('Planar path imported simulator modules: '+str(forbidden))
        print(json.dumps({'output':str(output),'backend':'planar_torch','simulator_modules_loaded':False}),flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
