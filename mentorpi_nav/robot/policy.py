"""Run the learned policy toward a supplied goal.

Sends actor commands directly through the sensor/heartbeat motor guard.
Requires fresh sensors and measured laser odometry.
Uses a fixed odometry goal and a fresh LiDAR-only occupancy grid.
"""
import argparse
import json
import math
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import torch
from .policy_io import PolicyIO
from .navigation_math import yaw_of
from .goals import add_arguments,validate
from .policy_runtime import warm_actor
from .motion_feedback import MotionProgressMonitor

from .paths import ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--tag',default='policy')
    args=parser.parse_args()
    validate(parser,args)
    if not args.tag.replace('_','').isalnum():parser.error('Simple log tag required')
    args.goal=None
    (ROOT/'diagnostics').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    actor=torch.jit.load(str(ROOT/'models/policy.ts')).eval()
    warmup=warm_actor(actor)
    print(json.dumps({'event':'POLICY_WARMED',**warmup}),flush=True)
    import fcntl
    session_lock=(ROOT/'policy_session.lock').open('w')
    try:fcntl.flock(session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another policy run is active. Stop it before starting a new goal.',flush=True)
        return 1
    rclpy.init();node=PolicyIO()
    laser=[]
    def receive(msg):
        laser[:]=[(time.monotonic(),msg)]
    node.create_subscription(Odometry,'/odom_rf2o',receive,10)
    result={'mode':'goal_policy','velocity_source':'installed_actor_only',
            'limits':{'seconds':args.seconds,'max_displacement_m':args.max_displacement,'linear_mps':.1,'yaw_radps':.3},
            'trace':[],'passed':False}
    result['warmup']=warmup
    result.update(goal_frame='odom',global_planner_used=False,obstacle_input='live_lidar')
    claimed=False
    try:
        if not node.ownership.wait_for_service(timeout_sec=20):raise RuntimeError('Ownership service unavailable')
        request=SetBool.Request();request.data=True
        future=node.ownership.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=3)
        if future.result() is None or not future.result().success:raise RuntimeError('Policy ownership denied')
        claimed=True;result['ownership_response']=future.result().message
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            node.phase(.05,(0.,0.,0.),False)
            if not node.health() and laser and node.status:break
        if node.health():raise RuntimeError('Readiness: '+','.join(node.health()))
        if not laser or time.monotonic()-laser[0][0]>.4:raise RuntimeError('Live measured laser odometry required')
        base,_=node.pose()
        if args.goal_relative is not None:
            forward,left=args.goal_relative
            c,s=math.cos(base[2]),math.sin(base[2])
            args.goal=[float(base[0]+c*forward-s*left),float(base[1]+s*forward+c*left)]
            result['requested_relative_goal']=args.goal_relative
        if args.goal_odom is not None:args.goal=args.goal_odom
        if args.goal is not None and np.linalg.norm(np.asarray(args.goal)-base[:2])<.15:
            result.update(passed=True,stop_reason='already_at_goal',goal=args.goal,goal_reached=True,
                          final_goal_distance_m=float(np.linalg.norm(np.asarray(args.goal)-base[:2])),
                          measured_pose_displacement_m=0.,measured_laser_displacement_m=0.)
            return 0
        goal=np.asarray(args.goal)
        start,_=node.pose();p=laser[0][1].pose.pose.position
        laser_start=np.array([p.x,p.y])
        result.update(start=start.tolist(),goal=goal.tolist(),laser_start=laser_start.tolist())
        result['start_enable_counts']=node.status.get('enable_counts')
        print(json.dumps({'event':'POLICY_START','goal':goal.tolist(),'seconds':args.seconds,
                          'max_displacement_m':args.max_displacement}),flush=True)
        previous=np.zeros(3);velocity=np.zeros(3)
        started=time.monotonic();stop_reason='time_limit'
        progress_monitor=MotionProgressMonitor()
        while args.seconds is None or time.monotonic()-started<args.seconds:
            health=node.health()
            if health:raise RuntimeError('Live health fault: '+','.join(health))
            if not laser or time.monotonic()-laser[0][0]>.4:raise RuntimeError('Measured laser odometry stale')
            base,_=node.pose()
            displacement=float(np.linalg.norm(base[:2]-start[:2]))
            p=laser[0][1].pose.pose.position
            laser_displacement=float(np.linalg.norm(np.array([p.x,p.y])-laser_start))
            q=laser[0][1].pose.pose.orientation
            progress_fault=progress_monitor.update(time.monotonic(),[p.x,p.y,yaw_of([q.x,q.y,q.z,q.w])],node.status.get('applied',[0.,0.,0.]))
            if progress_fault:raise RuntimeError(progress_fault)
            if args.max_displacement is not None and max(displacement,laser_displacement)>=args.max_displacement:
                stop_reason='displacement_limit';break
            if np.linalg.norm(goal-base[:2])<.15:
                stop_reason='goal_reached';break
            state,grid,details=node.observations(goal,previous)
            inference_started=time.perf_counter()
            with torch.no_grad():action=actor(torch.from_numpy(state),torch.from_numpy(grid)).numpy()[0]
            inference_ms=(time.perf_counter()-inference_started)*1000
            if inference_ms>100:raise RuntimeError('Inference exceeded 100 ms; no new motor command sent')
            drain_until=time.monotonic()+.01
            while time.monotonic()<drain_until:rclpy.spin_once(node,timeout_sec=0.)
            if node.health():raise RuntimeError('Post-inference health fault: '+','.join(node.health()))
            if args.seconds is not None and time.monotonic()-started>=args.seconds:
                stop_reason='time_limit';break
            if not np.isfinite(action).all():raise RuntimeError('Nonfinite policy action')
            action=np.clip(action,-1,1)
            desired=np.clip(action*np.array([.2,.2,.5]),[-.1,-.1,-.3],[.1,.1,.3])
            velocity+=np.clip(desired-velocity,[-.01,-.01,-.025],[.01,.01,.025])
            node.publish(velocity,True)
            result['trace'].append({'elapsed_s':time.monotonic()-started,'pose':base.tolist(),
                'laser_displacement_m':laser_displacement,'policy_action':action.tolist(),
                'requested_velocity':velocity.tolist(),'guard':node.status,
                'inference_ms':inference_ms,
                'front_obstacle_clearance_m':details['min_scan_clearance']})
            until=time.monotonic()+.05
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.005)
            previous=action
        result['stop_reason']=stop_reason
        node.phase(1,(0.,0.,0.),False)
        end,_=node.pose();p=laser[0][1].pose.pose.position
        result.update(end=end.tolist(),measured_pose_displacement_m=float(np.linalg.norm(end[:2]-start[:2])),
                      measured_laser_displacement_m=float(np.linalg.norm(np.array([p.x,p.y])-laser_start)),
                      final_goal_distance_m=float(np.linalg.norm(goal-end[:2])),
                      goal_reached=bool(np.linalg.norm(goal-end[:2])<.15),
                      controls=len(result['trace']),final_guard=node.status)
        result['passed']=bool(result['controls']>0 and result['measured_pose_displacement_m']>.02
                              and result['measured_laser_displacement_m']>.02)
    except Exception as exc:
        result.update(error=str(exc),stop_reason='fault')
    finally:
        node.phase(.5,(0.,0.,0.),False)
        result['final_guard']=node.status
        result['controls']=len(result['trace'])
        if 'start' in locals():
            try:
                final_pose,_=node.pose()
                result['end']=final_pose.tolist()
                result['measured_pose_displacement_m']=float(np.linalg.norm(final_pose[:2]-start[:2]))
                if laser:
                    final_point=laser[0][1].pose.pose.position
                    result['measured_laser_displacement_m']=float(np.linalg.norm(np.array([final_point.x,final_point.y])-laser_start))
            except Exception as exc:result['final_measurement_error']=str(exc)
        if claimed and node.ownership.service_is_ready():
            request=SetBool.Request();request.data=False
            future=node.ownership.call_async(request)
            rclpy.spin_until_future_complete(node,future,timeout_sec=3)
        (ROOT/'diagnostics'/f'{args.tag}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2),flush=True)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
