"""Train laser-sector TD3 using the shared planar/mecanum navigation model."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time

import torch
from mentorpi_nav.core import NavigationCfg,NavigationState
from mentorpi_nav.robot.laser_sectors import CONTRACT,OBS_SIZE
from mentorpi_nav.td3 import TD3,ReplayBuffer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario-file',default='scenarios/lidar_sector_td3.json')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--gpu-memory-fraction',type=float,default=1.,help='CUDA allocator cap as fraction of total VRAM')
    p.add_argument('--seed',type=int,default=72001)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--transitions',type=int,default=2000000)
    p.add_argument('--forever',action='store_true',help='Train until Ctrl+C; no transition limit')
    p.add_argument('--init-checkpoint',help='Continue weights/optimizers from TD3 checkpoint with fresh replay')
    p.add_argument('--random-steps',type=int,default=10000,help='Total transitions using uniformly random actions')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--replay-size',type=int,default=200000)
    p.add_argument('--updates-per-step',type=int,default=2)
    p.add_argument('--save-every',type=int,default=25000,help='Total transitions between checkpoints')
    p.add_argument('--keep-checkpoints',type=int,default=0,help='Keep latest N periodic saves; 0 keeps all')
    p.add_argument('--noise-start',type=float,default=.5)
    p.add_argument('--noise-end',type=float,default=.1)
    p.add_argument('--noise-decay-steps',type=int,default=500000)
    args=p.parse_args()
    if min(args.num_envs,args.transitions,args.batch_size,args.replay_size,args.updates_per_step,args.save_every,args.noise_decay_steps)<1:
        p.error('Counts must be positive')
    if args.replay_size<max(args.batch_size,args.num_envs) or args.random_steps<0:
        p.error('Invalid replay size or random warmup')
    if args.keep_checkpoints<0:p.error('Checkpoint retention must be nonnegative')
    if not 0<=args.noise_end<=args.noise_start<=1:p.error('Noise must satisfy 0<=end<=start<=1')
    if not 0<args.gpu_memory_fraction<=1:p.error('GPU memory fraction must be in (0,1]')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA requested but unavailable')
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,args.device)
    cfg=NavigationCfg(**json.loads(Path(args.scenario_file).read_text()))
    if cfg.policy_input!='laser_sectors':p.error('Use a sector-input scenario')
    source=None;source_saved=None
    if args.init_checkpoint:
        source=Path(args.init_checkpoint).resolve()
        prior=json.loads(source.with_name('task.json').read_text())
        previous=NavigationCfg(**prior['navigation'])
        if prior.get('network_architecture')!='laser_sector_td3_v1' or tuple(previous.speed_limits)!=tuple(cfg.speed_limits) or any(getattr(previous,k)!=getattr(cfg,k) for k in ('policy_input','translation_only','goal_normalization_half_extent')):
            p.error('Initialization requires matching observation/action contracts')
        source_saved=torch.load(source,map_location=args.device,weights_only=True)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'task.json').exists():p.error('Use a fresh output directory')
    manifest={'navigation':asdict(cfg),'model':'analytic_swept_disc_occupancy_v2',
              'network_architecture':'laser_sector_td3_v1','training_method':'td3','source_checkpoint':str(source) if source else None,
              'source_checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
              'source_transitions':source_saved.get('transitions') if source_saved else None,'fresh_replay':True,
              'observation_contract':CONTRACT,'observation_size':OBS_SIZE,'occupancy_shape':None,
              'seed':args.seed,'num_envs':args.num_envs,'backend':'planar_torch',
              'collision_footprint':'oriented_rectangle' if cfg.robot_size else 'disc',
              'reference':'https://github.com/reiniscimurs/DRL-robot-navigation',
              'adaptations':['36 body-frame 360-degree sectors','translation-only: yaw hard-masked to zero' if cfg.translation_only else 'three mecanum velocity actions',
                             'measured goal-direction velocity reward','footprint clearance penalty within0.2m']}
    (output/'task.json').write_text(json.dumps(manifest,indent=2))
    agent={'algorithm':'TD3','gamma':.995,'learning_rate':.0003,'tau':.005,'policy_noise':.2,
           'noise_clip':.5,'policy_delay':2,'hidden_dims':[256,256],**vars(args)}
    (output/'agent.json').write_text(json.dumps(agent,indent=2))
    state=NavigationState(args.num_envs,cfg,args.device,args.seed);state.reset()
    obs=state.observations()['policy']
    learner=TD3(args.device,translation_only=cfg.translation_only);replay=ReplayBuffer(args.replay_size,args.seed)
    if source_saved is not None:
        learner.restore(source_saved)
    learner.save(output/'model_0.pt',0)
    lengths=torch.zeros(args.num_envs,device=args.device,dtype=torch.long)
    returns=torch.zeros(args.num_envs,device=args.device)
    outcomes={'successes':0,'collisions':0,'timeouts':0};recent=deque(maxlen=100)
    transitions=0;next_save=args.save_every;next_log=1000;started=time.monotonic();metrics={}
    max_steps=round(cfg.episode_seconds/cfg.dt)
    try:
        while args.forever or transitions<args.transitions:
            noise=args.noise_start+(args.noise_end-args.noise_start)*min(1,transitions/args.noise_decay_steps)
            with torch.no_grad():
                if transitions<args.random_steps:action=2*torch.rand(args.num_envs,3,device=args.device)-1
                else:action=(learner.actor(obs)+torch.randn(args.num_envs,3,device=args.device)*noise).clamp(-1,1)
                if cfg.translation_only:action[:,2]=0
                state.step(action);lengths+=1;returns+=state.reward
                next_obs=state.observations()['policy']
                terminal=state.success|state.collision
                timeout=(lengths>=max_steps)&~terminal
                done=terminal|timeout
                # Store final observations BEFORE reset. Timeouts bootstrap;
                # arrivals and collisions do not.
                replay.add(obs,action,state.reward,next_obs,terminal)
                ids=done.nonzero().flatten()
                for i in ids.cpu().tolist():
                    kind='successes' if state.success[i] else ('collisions' if state.collision[i] else 'timeouts')
                    outcomes[kind]+=1;recent.append({'outcome':kind,'return':float(returns[i]),'seconds':float(lengths[i])*cfg.dt})
                if len(ids):
                    state.reset(ids);lengths[ids]=0;returns[ids]=0
                    next_obs=state.observations()['policy']
                obs=next_obs
            transitions+=args.num_envs
            if transitions>=args.random_steps and replay.size>=args.batch_size:
                for _ in range(args.updates_per_step):metrics=learner.update(replay.sample(args.batch_size,args.device))
            if transitions>=next_log:
                elapsed=time.monotonic()-started
                status={'transitions':transitions,'target_transitions':None if args.forever else args.transitions,'forever':args.forever,'updates':learner.updates,
                        'device':str(state.device),'actor_device':str(next(learner.actor.parameters()).device),
                        'elapsed_seconds':elapsed,'transitions_per_second':transitions/elapsed,
                        'eta_seconds':None if args.forever else max(0,args.transitions-transitions)*elapsed/transitions,'exploration_noise_std':noise,
                        'completed_episodes':sum(outcomes.values()),'outcomes':outcomes,
                        'recent_100':{kind:sum(r['outcome']==kind for r in recent) for kind in outcomes},
                        'recent_mean_return':sum(r['return'] for r in recent)/max(1,len(recent)),**metrics}
                temporary=output/'status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(output/'status.json')
                with (output/'metrics.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
                print(json.dumps(status),flush=True);next_log+=1000
            if transitions>=next_save:
                learner.save(output/f'model_{transitions}.pt',transitions);next_save+=args.save_every
                if args.keep_checkpoints:
                    saves=sorted(output.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
                    for old in saves[:-args.keep_checkpoints]:old.unlink()
        learner.save(output/'final.pt',transitions)
        print(json.dumps({'event':'TRAINING_COMPLETE','transitions':transitions,'outcomes':outcomes}),flush=True)
    except KeyboardInterrupt:
        learner.save(output/f'model_{transitions}.pt',transitions)
        print(json.dumps({'event':'INTERRUPTED','transitions':transitions}),flush=True)
    except Exception as exc:
        checkpoint=output/f'crash_{transitions}.pt'
        learner.save(checkpoint,transitions)
        print(json.dumps({'event':'TRAINING_ERROR','error':str(exc),'checkpoint':str(checkpoint),'transitions':transitions}),flush=True)
        raise


if __name__=='__main__':main()
