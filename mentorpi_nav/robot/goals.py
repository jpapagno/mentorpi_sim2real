"""Shared command-line goal and optional-limit arguments."""
import math

def add_arguments(parser):
    goals=parser.add_mutually_exclusive_group()
    goals.add_argument('--goal-relative',type=float,nargs=2,metavar=('FORWARD','LEFT'),help='Goal relative to the starting pose, in metres')
    goals.add_argument('--goal-odom',type=float,nargs=2,metavar=('X','Y'),help='Fixed odometry-frame goal, in metres')
    parser.add_argument('--seconds',type=float,help='Optional run time limit')
    parser.add_argument('--max-displacement',type=float,help='Optional displacement limit in metres')

def validate(parser,args):
    goal=args.goal_relative if args.goal_relative is not None else args.goal_odom
    if goal is None:parser.error('Supply --goal-relative FORWARD LEFT or --goal-odom X Y')
    if not all(math.isfinite(v) for v in goal):parser.error('Goal coordinates must be finite')
    for value in (args.seconds,args.max_displacement):
        if value is not None and (not math.isfinite(value) or value<=0):
            parser.error('Optional limits must be finite positive numbers')
