"""Run the policy toward a goal immediately; Ctrl+C stops navigation."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import uuid

ROOT=Path('/home/pi/docker/tmp/mentorpi_nav')
RUNTIME='/home/ubuntu/shared/mentorpi_nav/runtime/'

def ros(command,timeout=90,quiet=False,check=True):
    return subprocess.run(['docker','exec','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',
        'source /home/ubuntu/.zshrc >/dev/null; exec '+shlex.join(command)],check=check,timeout=timeout,
        capture_output=quiet,text=quiet)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Compatibility flag; supplying a goal already starts motion')
    parser.add_argument('--seconds',type=float,help='Optional run time limit in seconds')
    parser.add_argument('--max-displacement',type=float,help='Optional displacement limit in metres')
    goals=parser.add_mutually_exclusive_group()
    goals.add_argument('--goal',type=float,nargs=2,metavar=('X','Y'),help='Goal in saved-map metres')
    goals.add_argument('--goal-relative',type=float,nargs=2,metavar=('FORWARD','LEFT'),help='Goal in metres relative to the current robot pose')
    goals.add_argument('--goal-odom',type=float,nargs=2,metavar=('X','Y'),help='Goal in odometry-frame metres, without a saved map')
    parser.add_argument('--stop',action='store_true',help='Stop the navigation service and its motor bridge')
    args=parser.parse_args()
    if os.geteuid()!=0:parser.error('Run with sudo python3')
    if args.stop:
        subprocess.run(['systemctl','stop','mentorpi-navigation.service'],check=True)
        print('Navigation stopped; bridge teardown sends motor zero.')
        return
    for value in (args.seconds,args.max_displacement):
        if value is not None and (not math.isfinite(value) or value<=0):
            parser.error('Optional limits must be finite positive numbers')
    goal_values=args.goal if args.goal is not None else args.goal_relative if args.goal_relative is not None else args.goal_odom
    if goal_values is not None and not all(math.isfinite(v) for v in goal_values):parser.error('Goal coordinates must be finite')
    if goal_values is None:parser.error('Supply --goal-relative FORWARD LEFT or --goal-odom X Y (metres)')
    mode_file=ROOT/'config/localization_mode.json'
    mode=json.loads(mode_file.read_text()).get('mode') if mode_file.exists() else 'amcl'
    if args.goal is not None and mode=='local':parser.error('Saved-map --goal requires global mode. Use --goal-relative or --goal-odom for live local navigation.')
    subprocess.run(['systemctl','start','mentorpi-navigation.service'],check=True)
    print('Starting policy. Goal coordinates are metres. Ctrl+C stops the robot.',flush=True)
    tag='operator_policy_'+uuid.uuid4().hex[:8]
    goal_arguments=[]
    if args.goal is not None:goal_arguments=['--goal',*[str(v) for v in args.goal]]
    if args.goal_relative is not None:goal_arguments=['--goal-relative',*[str(v) for v in args.goal_relative]]
    if args.goal_odom is not None:goal_arguments=['--goal-odom',*[str(v) for v in args.goal_odom]]
    limits=[]
    if args.seconds is not None:limits+=['--seconds',str(args.seconds)]
    if args.max_displacement is not None:limits+=['--max-displacement',str(args.max_displacement)]
    try:
        ros(['python3',RUNTIME+'policy_commission.py','--tag',tag,*limits,*goal_arguments],timeout=None)
    finally:
        path=ROOT/'diagnostics'/(tag+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            print(json.dumps({k:result.get(k) for k in ('passed','stop_reason','error','controls',
                'goal','goal_reached','final_goal_distance_m','measured_pose_displacement_m',
                'measured_laser_displacement_m','final_guard')},indent=2))
            print('Full trace:',path)

if __name__=='__main__':
    try:main()
    except (KeyboardInterrupt,subprocess.TimeoutExpired):
        subprocess.run(['systemctl','stop','mentorpi-navigation.service'])
        print('Interrupted: navigation service stopped and motor-zero teardown requested.')
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print('Policy stopped before completing the goal. See the reason above.')
        raise SystemExit(exc.returncode or 1)
