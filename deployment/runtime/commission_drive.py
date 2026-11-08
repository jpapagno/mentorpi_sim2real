"""Bounded command-gate commissioning. Default mode never requests enabled motion."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool,String
from std_srvs.srv import SetBool
from navigation_io import NavigationIO


class CommissionDrive(NavigationIO):
    def __init__(self):
        super().__init__('mentorpi_drive_commission')
        self.command_pub=self.create_publisher(Twist,'/mentorpi/policy_cmd_vel',10)
        self.enable_pub=self.create_publisher(Bool,'/mentorpi/motion_enable',10)
        self.applied=[]
        self.status={}
        self.status_history=[]
        self.filtered=[]
        self.ownership=self.create_client(SetBool,'/mentorpi/commissioning_mode')
        self.create_subscription(Twist,'/mentorpi/applied_cmd_vel',self.on_applied,10)
        self.create_subscription(String,'/mentorpi/guard_status',self.on_status,10)
        self.create_subscription(Twist,'/mentorpi/guarded_cmd_vel',self.on_filtered,10)

    def on_applied(self,msg):self.applied.append((time.monotonic(),msg.linear.x,msg.linear.y,msg.angular.z))
    def on_filtered(self,msg):self.filtered.append((time.monotonic(),msg.linear.x,msg.linear.y,msg.angular.z))
    def on_status(self,msg):
        self.status=json.loads(msg.data)
        self.status_history.append((time.monotonic(),self.status))

    def publish(self,velocity,enabled):
        msg=Twist();msg.linear.x=float(velocity[0]);msg.linear.y=float(velocity[1]);msg.angular.z=float(velocity[2])
        enable=Bool();enable.data=enabled
        self.enable_pub.publish(enable);self.command_pub.publish(msg)

    def phase(self,seconds,velocity,enabled,publish_command=True,require_health=False):
        started=time.monotonic()
        while time.monotonic()-started<seconds:
            if require_health:
                reasons=self.health()
                if reasons:raise RuntimeError('Commissioning sensor fault: '+','.join(reasons))
            if publish_command:self.publish(velocity,enabled)
            else:
                message=Bool();message.data=enabled;self.enable_pub.publish(message)
            until=time.monotonic()+.05
            while time.monotonic()<until:rclpy.spin_once(self,timeout_sec=.005)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pulse',choices=['x','y','yaw'])
    parser.add_argument('--speed',type=float,default=.03)
    parser.add_argument('--duration',type=float,default=1.)
    args=parser.parse_args()
    if not 0<args.speed<=.1 or not 0<args.duration<=2. or args.speed*args.duration>.06:raise ValueError('Commissioning pulse exceeds bounds')
    rclpy.init();node=CommissionDrive()
    result={'mode':args.pulse or 'stationary_watchdog','stages':[]}
    try:
        if node.ownership.wait_for_service(timeout_sec=2.):
            request=SetBool.Request();request.data=True
            future=node.ownership.call_async(request)
            rclpy.spin_until_future_complete(node,future,timeout_sec=3.)
            if future.result() is None or not future.result().success:raise RuntimeError('Commissioning ownership denied')
        node.phase(10.,(0.,0.,0.),False)
        if node.health():raise RuntimeError(str(node.health()))
        start,_=node.pose()
        at=time.monotonic()
        node.phase(.7,(.03,0.,0.),False)
        disarmed=[row[1:] for row in node.applied if row[0]>=at]
        if not disarmed or np.max(np.abs(disarmed))>1e-7:raise RuntimeError('Disarmed nonzero command was applied')
        result['stages'].append('nonzero_input_blocked_while_disarmed')
        node.phase(.5,(0.,0.,0.),True)
        node.phase(.65,(0.,0.,0.),True,publish_command=False)
        if 'command_stale' not in node.status.get('blocked_by',[]):raise RuntimeError('Command timeout not enforced')
        result['stages'].append('command_timeout_stops')
        node.phase(.5,(0.,0.,0.),False)
        result['stages'].append('explicit_disarm_stops')
        if args.pulse:
            velocity=[0.,0.,0.];velocity[['x','y','yaw'].index(args.pulse)]=args.speed
            before,_=node.pose()
            pulse_at=time.monotonic()
            node.phase(args.duration,velocity,True,require_health=True)
            pulse_end=time.monotonic()
            node.phase(1.,(0.,0.,0.),False)
            after,_=node.pose()
            result.update(before=before.tolist(),after=after.tolist(),pose_delta=(after-before).tolist(),requested_velocity=velocity,duration=args.duration)
            applied=[row[1:] for row in node.applied if pulse_at<=row[0]<=pulse_end]
            result['pulse_applied_nonzero_samples']=sum(any(abs(v)>1e-6 for v in row) for row in applied)
            result['pulse_filtered_nonzero_samples']=sum(any(abs(v)>1e-6 for v in row[1:]) for row in node.filtered if pulse_at<=row[0]<=pulse_end)
            result['pulse_block_reasons']=sorted({reason for stamp,status in node.status_history if pulse_at<=stamp<=pulse_end for reason in status.get('blocked_by',[])})
            delta=after-before
            if args.pulse=='yaw':measured=float(delta[2])
            else:
                c,s=np.cos(before[2]),np.sin(before[2])
                measured=float(c*delta[0]+s*delta[1]) if args.pulse=='x' else float(-s*delta[0]+c*delta[1])
            result['measured_along_requested_axis']=measured
            if result['pulse_applied_nonzero_samples']==0 or measured<.005:
                raise RuntimeError('Drive pulse was blocked or produced insufficient measured motion')
        end,_=node.pose()
        result.update(passed=True,start=start.tolist(),end=end.tolist(),guard_status=node.status)
    except Exception as exc:
        result.update(passed=False,error=str(exc))
    finally:
        node.phase(.5,(0.,0.,0.),False)
        if node.ownership.service_is_ready():
            request=SetBool.Request();request.data=False
            future=node.ownership.call_async(request)
            rclpy.spin_until_future_complete(node,future,timeout_sec=3.)
        target=Path('/home/ubuntu/shared/mentorpi_nav/diagnostics')/('commission_'+(args.pulse or 'stationary')+'.json')
        target.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    if not result.get('passed'):raise SystemExit(1)


if __name__=='__main__':main()
