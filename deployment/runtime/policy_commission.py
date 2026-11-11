"""Run the learned policy toward a supplied goal.

Sends actor commands directly through the sensor/heartbeat motor guard.
Requires fresh sensors and measured laser odometry.
Local mode consumes a direct odometry-frame goal and a fresh sensor-only grid;
legacy map mode uses localization and a Nav2 path. No scripted drive fallback.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
from nav2_msgs.action import ComputePathToPose
from std_srvs.srv import SetBool
import torch
from commission_drive import CommissionDrive
from navigation_math import reachable_goal_candidates,subgoal_on_path,yaw_of
from policy_runtime import warm_actor
from motion_feedback import MotionProgressMonitor

ROOT=Path('/home/ubuntu/shared/mentorpi_nav')

def plan(node, client, goal):
    request=ComputePathToPose.Goal()
    request.goal.header.frame_id='map'
    request.goal.header.stamp=node.get_clock().now().to_msg()
    request.goal.pose.position.x=float(goal[0]);request.goal.pose.position.y=float(goal[1])
    request.goal.pose.orientation.w=1.
    request.planner_id='GridBased';request.use_start=False
    future=client.send_goal_async(request)
    rclpy.spin_until_future_complete(node,future,timeout_sec=5)
    handle=future.result()
    if handle is None or not handle.accepted:return None
    future=handle.get_result_async()
    rclpy.spin_until_future_complete(node,future,timeout_sec=5)
    response=future.result()
    if response is None or response.status!=4 or len(response.result.path.poses)<2:return None
    return np.asarray([[p.pose.position.x,p.pose.position.y] for p in response.result.path.poses])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,help='Optional time limit')
    parser.add_argument('--max-displacement',type=float,help='Optional displacement limit')
    parser.add_argument('--tag',default='policy_commission')
    goals=parser.add_mutually_exclusive_group()
    goals.add_argument('--goal',type=float,nargs=2,help='Optional map-frame goal; still checked by Nav2')
    goals.add_argument('--goal-relative',type=float,nargs=2,metavar=('FORWARD','LEFT'),help='Goal in metres relative to the current robot pose')
    goals.add_argument('--goal-odom',type=float,nargs=2,metavar=('X','Y'),help='Goal in odometry-frame metres, without a saved map')
    args=parser.parse_args()
    for value in (args.seconds,args.max_displacement):
        if value is not None and (not math.isfinite(value) or value<=0):
            raise ValueError('Optional limits must be finite positive numbers')
    if not args.tag.replace('_','').isalnum():raise ValueError('Simple log tag required')
    if args.goal is not None and not np.isfinite(args.goal).all():raise ValueError('Finite goal coordinates required')
    if args.goal_relative is not None and not np.isfinite(args.goal_relative).all():raise ValueError('Finite relative goal coordinates required')
    if args.goal_odom is not None and not np.isfinite(args.goal_odom).all():raise ValueError('Finite odometry goal coordinates required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    actor=torch.jit.load(str(ROOT/'models/policy.ts')).eval()
    warmup=warm_actor(actor)
    print(json.dumps({'event':'POLICY_WARMED',**warmup}),flush=True)
    import fcntl
    session_lock=(ROOT/'policy_session.lock').open('w')
    fcntl.flock(session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    local_mode=True if args.goal_relative is not None or args.goal_odom is not None else False if args.goal is not None else None
    rclpy.init();node=CommissionDrive(local_mode=local_mode)
    client=None if node.local_mode else ActionClient(node,ComputePathToPose,'/compute_path_to_pose')
    laser=[]
    def receive(msg):
        laser[:]=[(time.monotonic(),msg)]
    node.create_subscription(Odometry,'/odom_rf2o',receive,10)
    result={'mode':'goal_policy','velocity_source':'installed_actor_only',
            'limits':{'seconds':args.seconds,'max_displacement_m':args.max_displacement,'linear_mps':.1,'yaw_radps':.3},
            'trace':[],'passed':False}
    result['warmup']=warmup
    result.update(goal_frame=node.pose_frame,global_planner_used=not node.local_mode,
                  obstacle_input='live_lidar' if node.local_mode else 'saved_map_plus_lidar')
    claimed=False
    try:
        if not node.ownership.wait_for_service(timeout_sec=20):raise RuntimeError('Ownership service unavailable')
        request=SetBool.Request();request.data=True
        future=node.ownership.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=3)
        if future.result() is None or not future.result().success:raise RuntimeError('Commissioning ownership denied')
        claimed=True;result['ownership_response']=future.result().message
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            node.phase(.05,(0.,0.,0.),False)
            if not node.health() and laser and node.status:break
        if node.health():raise RuntimeError('Readiness: '+','.join(node.health()))
        if not laser or time.monotonic()-laser[0][0]>.4:raise RuntimeError('Live measured laser odometry required')
        if client is not None and not client.wait_for_server(timeout_sec=3):raise RuntimeError('Planner unavailable')
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
        path=None
        if node.local_mode:
            if args.goal is None:raise RuntimeError('Supply --goal-relative or --goal-odom for local navigation')
            goal=np.asarray(args.goal)
        else:
            origin=node.map.info.origin;q=origin.orientation
            values=np.asarray(node.map.data).reshape(node.map.info.height,node.map.info.width)
            candidates=reachable_goal_candidates(values,node.map.info.resolution,
                [origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])],base,
                clearance=.28,min_distance=.2 if args.goal else .5,max_distance=1. if args.goal else .75)
            if not len(candidates):raise RuntimeError('No footprint-cleared local goal')
            delta=candidates-base[:2]
            forward=np.cos(base[2])*delta[:,0]+np.sin(base[2])*delta[:,1]
            left=-np.sin(base[2])*delta[:,0]+np.cos(base[2])*delta[:,1]
            order=np.argsort(np.linalg.norm(candidates-np.asarray(args.goal),axis=1)) if args.goal else np.argsort(abs(np.arctan2(left,-forward))+.2*np.linalg.norm(delta,axis=1))
            planning_started=time.monotonic()
            for index in order[:12]:
                if time.monotonic()-planning_started>15:break
                if args.goal is None and forward[index]>-.3:continue
                if args.goal is not None and np.linalg.norm(candidates[index]-np.asarray(args.goal))>.1:continue
                goal=np.asarray(args.goal) if args.goal is not None else candidates[index]
                path=plan(node,client,goal)
                if path is not None:break
            if path is None:raise RuntimeError('No Nav2-verified goal')
        if node.health():raise RuntimeError('Readiness changed during planning: '+','.join(node.health()))
        start,_=node.pose();p=laser[0][1].pose.pose.position
        laser_start=np.array([p.x,p.y])
        result.update(start=start.tolist(),goal=goal.tolist(),path=path.tolist() if path is not None else [],laser_start=laser_start.tolist())
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
            target=goal if node.local_mode else subgoal_on_path(path,base,.5)
            state,grid,details=node.observations(target,previous)
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
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','path')},indent=2),flush=True)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
