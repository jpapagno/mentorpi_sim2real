"""Supervised bounded body-motion calibration; no hardware access at import."""
import argparse
import json
import math
from pathlib import Path
import time


def body_delta(start,end):
    dx,dy=end[0]-start[0],end[1]-start[1]
    c,s=math.cos(start[2]),math.sin(start[2])
    return [c*dx+s*dy,-s*dx+c*dy,(end[2]-start[2]+math.pi)%(2*math.pi)-math.pi]


def measurable_response(axis,delta):
    return bool(abs(delta[axis]) >= (.02 if axis==2 else .01))


def displacement_exceeded(delta,settling=False):
    return math.hypot(*delta[:2])>(.05 if settling else .12) or abs(delta[2])>(.12 if settling else .5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Run operator-authorized calibration pulses')
    p.add_argument('--output',required=True)
    p.add_argument('--linear-speed',type=float,default=.06)
    p.add_argument('--yaw-speed',type=float,default=.2)
    p.add_argument('--pulse-seconds',type=float,default=1.5)
    p.add_argument('--repetitions',type=int,choices=[1,2],default=2)
    p.add_argument('--directions',nargs='+',choices=['forward','backward','left','right','ccw','cw'],default=['forward','backward','left','right','ccw','cw'])
    p.add_argument('--allow-stationary',action='store_true',help='Record stationary response in short threshold tests instead of fitting a gain')
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--check-clearance',action='store_true',help='Require observed enclosing-circle clearance during diagnostic motion')
    args=p.parse_args()
    if not args.execute:p.error('Motion requires explicit --execute')
    if not (0<args.linear_speed<=.1 and 0<args.yaw_speed<=.3 and 0<args.pulse_seconds<=1.5 and args.linear_speed*args.pulse_seconds<=.100001):
        p.error('Pulse exceeds bounded calibration limits')
    if args.allow_stationary and args.pulse_seconds>1:
        p.error('Stationary threshold tests are limited to one second')
    if not 1<=args.settle_seconds<=10:p.error('Settling hold must be1..10seconds')
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from std_srvs.srv import SetBool
    from .policy_io import PolicyIO
    from .motion_profile import calibration_path_clear
    rclpy.init();node=PolicyIO();laser=[];imu=[];guard=[];rows=[]
    result={'passed':False,'translation_mps':args.linear_speed,'yaw_radps':args.yaw_speed,'pulse_seconds':args.pulse_seconds,'repetitions':args.repetitions,'pulses':[]}
    claimed=False
    def on_laser(msg):
        from .navigation_math import yaw_of
        q=msg.pose.pose.orientation;pos=msg.pose.pose.position
        laser[:]=[(time.monotonic(),[pos.x,pos.y,yaw_of([q.x,q.y,q.z,q.w])])]
    node.create_subscription(Odometry,'/odom_rf2o',on_laser,10)
    node.create_subscription(Imu,'/imu',lambda m:imu.append((time.monotonic(),float(m.angular_velocity.z))),10)
    def ownership(value):
        req=SetBool.Request();req.data=value
        future=node.ownership.call_async(req);rclpy.spin_until_future_complete(node,future,timeout_sec=3)
        if value and (future.result() is None or not future.result().success):raise RuntimeError('Ownership denied')
    def collect(seconds,velocity,enabled,origin=None,settling=False):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            health=node.health()
            if enabled and health:raise RuntimeError('Sensor health: '+','.join(health))
            if enabled and (not laser or time.monotonic()-laser[0][0]>.4):raise RuntimeError('Laser odometry stale')
            if origin is not None:
                if enabled and args.check_clearance and not calibration_path_clear(node.scan_points()[:,:2].tolist(),[0.,0.]):
                    raise RuntimeError('Diagnostic obstacle-clearance stop')
                pose,_=node.pose();delta=body_delta(origin,pose)
                if displacement_exceeded(delta,settling):raise RuntimeError('Settling displacement limit' if settling else 'Pulse displacement limit')
                global_delta=body_delta(session_origin,pose)
                if math.hypot(*global_delta[:2])>.3 or abs(global_delta[2])>1.:raise RuntimeError('Session displacement limit')
            node.publish(velocity,enabled)
            until=min(deadline,time.monotonic()+.05)
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.005)
            if node.status:
                pose,_=node.pose()
                rows.append({'time':time.monotonic(),'pose':pose.tolist(),'laser':laser[0][1] if laser else None,
                    'requested':list(velocity),'enabled':enabled,'guard':dict(node.status)})
    try:
        if not node.ownership.wait_for_service(timeout_sec=10):raise RuntimeError('Ownership unavailable')
        ownership(True);claimed=True
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            node.phase(.1,(0.,0.,0.),False)
            if not node.health() and laser and node.status:break
        if node.health() or not laser:raise RuntimeError('Sensors not ready: '+','.join(node.health()))
        session_origin,_=node.pose();result['start']=session_origin.tolist()
        collect(1.,(0.,0.,0.),False)
        gyro_baseline=sum(v for _,v in imu[-10:])/max(1,len(imu[-10:]))
        result['stationary_gyro_z_bias_radps']=gyro_baseline
        linear,yaw=args.linear_speed,args.yaw_speed
        commands={'forward':(linear,0.,0.),'backward':(-linear,0.,0.),
                  'left':(0.,linear,0.),'right':(0.,-linear,0.),
                  'ccw':(0.,0.,yaw),'cw':(0.,0.,-yaw)}
        for repeat in range(args.repetitions):
            for name in args.directions:
                velocity=commands[name]
                origin,_=node.pose();laser_origin=laser[0][1][:];start=time.monotonic();index=len(rows)
                print(json.dumps({'event':'PULSE_START','direction':name,'repeat':repeat+1}),flush=True)
                collect(args.pulse_seconds,velocity,True,origin)
                command_end=time.monotonic();stop_pose,_=node.pose()
                collect(args.settle_seconds,(0.,0.,0.),False,stop_pose,settling=True)
                end,_=node.pose();laser_end=laser[0][1][:]
                pulse_rows=rows[index:];integral=[0.,0.,0.];gyro_angle=0.
                for a,b in zip(pulse_rows,pulse_rows[1:]):
                    dt=b['time']-a['time']
                    for k,value in enumerate(a['guard'].get('applied',[0.,0.,0.])):integral[k]+=dt*value
                gyro=[(stamp,v) for stamp,v in imu if start<=stamp<=time.monotonic()]
                for a,b in zip(gyro,gyro[1:]):gyro_angle+=(a[1]-gyro_baseline)*(b[0]-a[0])
                measured=body_delta(origin,end);axis=2 if name in ('ccw','cw') else (1 if name in ('left','right') else 0)
                row={'direction':name,'repeat':repeat+1,'requested_velocity':velocity,'command_duration_s':command_end-start,
                    'start_time':start,'command_end_time':command_end,'stop_pose':stop_pose.tolist(),
                    'body_delta':measured,'laser_body_delta':body_delta(laser_origin,laser_end),
                    'settling_body_delta':body_delta(stop_pose,end),'response_measurable':measurable_response(axis,measured),
                    'integrated_applied_velocity':integral,'gyro_angle_radians':gyro_angle,
                    'axis_gain':measured[axis]/integral[axis] if abs(integral[axis])>1e-4 else None,
                    'final_guard':dict(node.status)}
                result['pulses'].append(row);print(json.dumps({'event':'PULSE_RESULT',**row}),flush=True)
                if abs(integral[axis])<.005:raise RuntimeError('Pulse blocked or insufficient applied command')
                if measured[axis]*(1 if velocity[axis]>0 else -1)<-0.003:raise RuntimeError('Measured motion opposes requested axis')
                if not measurable_response(axis,measured) and not args.allow_stationary:raise RuntimeError('Insufficient measured response; calibration invalid, stop before further pulses')
        result['passed']=True
    except Exception as exc:result['error']=str(exc)
    finally:
        if claimed:
            node.phase(.8,(0.,0.,0.),False)
            if node.ownership.service_is_ready():ownership(False)
        result['final_guard']=node.status;result['trace']=rows;result['gyro_samples']=imu
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','pulses')}),flush=True)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
