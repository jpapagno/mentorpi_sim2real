"""Start a local-goal policy run in the configured ROS container."""
import argparse
import os
import shlex
import subprocess
import uuid
from .goals import add_arguments,validate
from .paths import ROOT

CONTAINER_ROOT='/home/ubuntu/shared/mentorpi_nav'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--stop',action='store_true',help='Stop navigation and motor output')
    args=parser.parse_args()
    if not args.stop:validate(parser,args)
    if getattr(os,'geteuid',lambda: -1)()!=0:parser.error('Run with sudo python3 on the Pi')
    service=['systemctl','stop' if args.stop else 'start','mentorpi-navigation.service']
    subprocess.run(service,check=True)
    if args.stop:return 0
    tag='operator_policy_'+uuid.uuid4().hex[:8]
    command=['python3','-m','mentorpi_nav.robot.policy','--tag',tag]
    if args.goal_relative is not None:command+=['--goal-relative',*map(str,args.goal_relative)]
    else:command+=['--goal-odom',*map(str,args.goal_odom)]
    if args.seconds is not None:command+=['--seconds',str(args.seconds)]
    if args.max_displacement is not None:command+=['--max-displacement',str(args.max_displacement)]
    print('Starting policy. Goal coordinates are metres. Ctrl+C stops the robot.',flush=True)
    try:
        result=subprocess.run(['docker','exec','-u','ubuntu','-w',CONTAINER_ROOT,'MentorPi','/bin/zsh','-c',
            'source /home/ubuntu/.zshrc >/dev/null; exec '+shlex.join(command)])
        print('Full trace:',ROOT/'diagnostics'/(tag+'.json'))
        return result.returncode
    except KeyboardInterrupt:
        subprocess.run(['systemctl','stop','mentorpi-navigation.service'])
        print('Navigation stopped.')
        return 130

if __name__=='__main__':raise SystemExit(main())
