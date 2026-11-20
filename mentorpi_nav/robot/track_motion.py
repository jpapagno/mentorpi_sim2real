"""Fixed feedback controller for supervised calibration targets; preview default.

No RL, no hardware access at import, no autonomous profile updates. A profile
changes command compensation only. This entry point has not been validated
on the physical robot; post-motion buzzing is unresolved.
"""
import argparse
import json
import math
from pathlib import Path
import time
from .motion_profile import validate_profile,identity_profile,fixed_controller,calibration_path_clear


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',nargs=3,type=float,required=True,metavar=('FORWARD_M','LEFT_M','YAW_DEG'))
    p.add_argument('--profile',help='Measured profile JSON; omit for identity baseline')
    p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    forward,left,degrees=args.target
    if not all(math.isfinite(v) for v in args.target) or math.hypot(forward,left)>.12 or abs(degrees)>15:
        p.error('Calibration target limited to12cm and15degrees')
    profile=validate_profile(json.loads(Path(args.profile).read_text())) if args.profile else identity_profile()
    result={'target_relative':args.target,'profile':args.profile or 'identity','controller':'fixed_proportional_v1',
        'gains':[1.,1.,1.5],'position_tolerance_m':.01,'yaw_tolerance_degrees':2.,'deadline_seconds':10,
        'passed':False,'hardware_validated':False,'trace':[]}
    if not args.execute:
        print(json.dumps({**result,'preview':True,'nonzero_commands_sent':0},indent=2));return 0
    import fcntl
    import rclpy
    from std_srvs.srv import SetBool
    from .paths import ROOT
    from .policy_io import PolicyIO
    lock=(ROOT/'policy_session.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init();node=PolicyIO();claimed=False
    def ownership(value):
        req=SetBool.Request();req.data=value
        future=node.ownership.call_async(req);rclpy.spin_until_future_complete(node,future,timeout_sec=3)
        if value and (future.result() is None or not future.result().success):raise RuntimeError('Ownership denied')
    def errors(pose):
        dx,dy=goal[0]-pose[0],goal[1]-pose[1];c,s=math.cos(pose[2]),math.sin(pose[2])
        return [c*dx+s*dy,-s*dx+c*dy],(goal[2]-pose[2]+math.pi)%(2*math.pi)-math.pi
    try:
        if not node.ownership.wait_for_service(timeout_sec=10):raise RuntimeError('Ownership unavailable')
        ownership(True);claimed=True
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            node.phase(.1,(0.,0.,0.),False)
            if not node.health() and node.status:break
        if node.health():raise RuntimeError('Sensor health:'+','.join(node.health()))
        start,_=node.pose();c,s=math.cos(start[2]),math.sin(start[2])
        if not calibration_path_clear(node.scan_points()[:,:2].tolist(),[forward,left]):
            raise RuntimeError('Diagnostic target lacks observed obstacle clearance')
        goal=[start[0]+c*forward-s*left,start[1]+s*forward+c*left,start[2]+math.radians(degrees)]
        result.update(start=start.tolist(),goal=goal);began=time.monotonic();settled_since=None
        progress_origin=start.copy();progress_at=began
        while time.monotonic()-began<10:
            if node.health():raise RuntimeError('Sensor health:'+','.join(node.health()))
            pose,_=node.pose();body_error,yaw_error=errors(pose)
            if not calibration_path_clear(node.scan_points()[:,:2].tolist(),[0.,0.],margin=.01):
                raise RuntimeError('Diagnostic obstacle-clearance stop')
            if math.hypot(pose[0]-start[0],pose[1]-start[1])>.18 or abs((pose[2]-start[2]+math.pi)%(2*math.pi)-math.pi)>.6:
                raise RuntimeError('Calibration displacement bound')
            command,reached=fixed_controller(body_error,yaw_error,profile)
            if reached:
                node.publish((0.,0.,0.),False)
                progress_origin=pose.copy();progress_at=time.monotonic()
                if settled_since is None:settled_since=time.monotonic()
                if time.monotonic()-settled_since>=.8:
                    result['stop_reason']='settled_target';break
            else:
                settled_since=None;node.publish(command,True)
                yaw_progress=abs((pose[2]-progress_origin[2]+math.pi)%(2*math.pi)-math.pi)
                xy_progress=math.hypot(pose[0]-progress_origin[0],pose[1]-progress_origin[1])
                if xy_progress>.002 or yaw_progress>math.radians(.5):
                    progress_origin=pose.copy();progress_at=time.monotonic()
                elif time.monotonic()-progress_at>.8:
                    raise RuntimeError('No measured response; stop short of prolonged motor buzzing')
            result['trace'].append({'time':time.monotonic(),'pose':pose.tolist(),'body_error':body_error,
                'yaw_error':yaw_error,'command':command,'enabled':not reached,'guard':node.status})
            until=time.monotonic()+.05
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.005)
        else:result['stop_reason']='time_limit'
    except Exception as exc:result.update(stop_reason='fault',error=str(exc))
    finally:
        if claimed:
            node.phase(1.,(0.,0.,0.),False)
            try:
                pose,_=node.pose();xy,yaw=errors(pose)
                result.update(end=pose.tolist(),final_position_error_m=math.hypot(*xy),final_yaw_error_degrees=math.degrees(yaw))
                result['passed']=result.get('stop_reason')=='settled_target' and math.hypot(*xy)<=.01 and abs(yaw)<=math.radians(2)
            except Exception as exc:result['final_measurement_error']=str(exc)
            if node.ownership.service_is_ready():ownership(False)
        result['final_guard']=node.status
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2))
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
