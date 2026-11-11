"""Onboard learned local controller and bounded automatically sampled trials.

Starts disarmed. /mentorpi/start_trials explicitly arms one 30-goal session.
Global paths come from Nav2; every local velocity comes from the learned actor.
"""
import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool,String
from std_srvs.srv import Trigger,SetBool
from nav2_msgs.action import ComputePathToPose
import torch

from navigation_io import NavigationIO
from navigation_math import reachable_goal_candidates,subgoal_on_path,yaw_of
from policy_runtime import warm_actor
from motion_feedback import MotionProgressMonitor


class AutonomousTrials(NavigationIO):
    def __init__(self,episodes=30,seed=2026,goal_seconds=90.):
        super().__init__('mentorpi_autonomous_trials')
        self.root=Path('/home/ubuntu/shared/mentorpi_nav')
        self.actor=torch.jit.load(str(self.root/'models/policy.ts')).eval()
        self.inference_warmup=warm_actor(self.actor)
        self.motion_progress=MotionProgressMonitor()
        (self.root/'diagnostics/policy_warmup.json').write_text(json.dumps(self.inference_warmup,indent=2))
        self.planner=ActionClient(self,ComputePathToPose,'/compute_path_to_pose')
        self.command_pub=self.create_publisher(Twist,'/mentorpi/policy_cmd_vel',10)
        self.enable_pub=self.create_publisher(Bool,'/mentorpi/motion_enable',10)
        self.status_pub=self.create_publisher(String,'/mentorpi/navigation_status',10)
        self.create_service(Trigger,'/mentorpi/start_trials',self.start)
        self.create_service(Trigger,'/mentorpi/stop_trials',self.stop_service)
        self.create_service(SetBool,'/mentorpi/commissioning_mode',self.commissioning_mode)
        self.create_subscription(String,'/mentorpi/guard_status',self.guard_status,10)
        self.create_subscription(Twist,'/mentorpi/guarded_cmd_vel',self.filtered_command,10)
        self.episodes=episodes;self.seed=seed;self.rng=np.random.default_rng(seed)
        self.goal_seconds=goal_seconds
        self.active=False;self.mode='DISARMED';self.previous_action=np.zeros(3,dtype=np.float32)
        self.last_velocity=np.zeros(3);self.filtered_velocity=np.zeros(3);self.guard={}
        self.goal=None;self.path=None;self.goal_started=0.;self.session_started=0.
        self.results=[];self.rejected_goals=[];self.settled_at=None;self.plan_deadline=0.
        self.session_id=None;self.sample_attempts=0;self.intervention_started=None
        self.last_status=0.;self.last_replan=0.;self.plan_pending=False;self.last_health=[]
        self.trace=None
        self.plan_generation=0
        self.commissioning_paused=False
        self.create_timer(.05,self.tick)

    def guard_status(self,msg):self.guard=json.loads(msg.data)
    def filtered_command(self,msg):self.filtered_velocity=np.array([msg.linear.x,msg.linear.y,msg.angular.z])

    def start(self,request,response):
        if self.commissioning_paused:
            response.success=False;response.message='Commissioning mode is active';return response
        if self.active:
            response.success=False;response.message='A trial session is already active';return response
        health=self.health()
        acceptance=self.root/'diagnostics/commissioning_acceptance.json'
        if health or not acceptance.exists():
            response.success=False;response.message='Not armed: '+(','.join(health) if health else 'physical commissioning not accepted');return response
        if not json.loads(acceptance.read_text()).get('passed'):
            response.success=False;response.message='Physical commissioning not accepted';return response
        if not self.planner.server_is_ready():
            response.success=False;response.message='Nav2 planner is not ready';return response
        self.session_id=uuid.uuid4().hex[:12]
        self.session_started=time.monotonic();self.results=[];self.rejected_goals=[]
        self.rng=np.random.default_rng(self.seed);self.sample_attempts=0
        self.trace=(self.root/'logs'/f'trials_{self.session_id}.jsonl').open('w',buffering=1)
        self.active=True;self.mode='SELECT';self.goal=None;self.path=None
        self.motion_progress.start=None
        self.previous_action[:]=0;self.last_velocity[:]=0
        response.success=True;response.message=f'Started {self.episodes} automatic goals, session {self.session_id}'
        self.event('session_start',seed=self.seed,episodes=self.episodes)
        return response

    def stop_service(self,request,response):
        self.stop('explicit_stop');response.success=True;response.message='Stopped and disarmed';return response

    def commissioning_mode(self,request,response):
        if self.active:
            response.success=False;response.message='Stop the trial session before commissioning';return response
        self.publish([0.,0.,0.],False)
        self.commissioning_paused=request.data
        self.get_logger().info('Commissioning ownership paused='+str(self.commissioning_paused))
        response.success=True;response.message='Commissioning controller ownership' if request.data else 'Returned to disarmed trial controller'
        return response

    def event(self,name,**fields):
        record={'event':name,'elapsed_s':time.monotonic()-self.session_started,**fields}
        if self.trace:self.trace.write(json.dumps(record)+'\n')

    def publish(self,velocity,enabled):
        command=Twist();command.linear.x=float(velocity[0]);command.linear.y=float(velocity[1]);command.angular.z=float(velocity[2])
        arm=Bool();arm.data=enabled
        self.enable_pub.publish(arm);self.command_pub.publish(command)

    def report(self,reason):
        successes=sum(row['outcome']=='success' for row in self.results)
        result={'session_id':self.session_id,'seed':self.seed,'requested_goals':self.episodes,
            'completed_goals':len(self.results),'successes':successes,
            'timeouts':sum(row['outcome']=='timeout' for row in self.results),
            'faults':sum(row['outcome']=='fault' for row in self.results),
            'success_rate':successes/len(self.results) if self.results else None,
            'complete':len(self.results)==self.episodes,'stop_reason':reason,
            'elapsed_s':time.monotonic()-self.session_started,'trials':self.results,
            'rejected_goal_samples':self.rejected_goals,
            'physical_contact_verification':'requires supervisor/camera review; no bumper sensor',
            'clock_note':'Durations use monotonic time; robot wall clock is not synchronized to laptop'}
        (self.root/'logs'/f'trials_{self.session_id}_summary.json').write_text(json.dumps(result,indent=2))
        (self.root/'logs/latest_trials.json').write_text(json.dumps(result,indent=2))

    def stop(self,reason):
        was_active=self.active
        self.active=False;self.mode='DISARMED';self.publish([0.,0.,0.],False)
        self.plan_generation+=1;self.plan_pending=False
        self.previous_action[:]=0;self.last_velocity[:]=0
        if was_active:
            self.event('session_stop',reason=reason)
            self.report(reason)
            if self.trace:self.trace.close();self.trace=None

    def finish_goal(self,outcome,**details):
        base,_=self.pose()
        row={'index':len(self.results)+1,'goal':self.goal.tolist(),'outcome':outcome,
            'duration_s':time.monotonic()-self.goal_started,'final_pose':base.tolist(),
            'final_distance':float(np.linalg.norm(self.goal-base[:2])),**details}
        self.results.append(row);self.event('goal_finished',**row)
        self.plan_generation+=1;self.plan_pending=False
        self.goal=None;self.path=None;self.settled_at=None;self.previous_action[:]=0;self.last_velocity[:]=0
        self.publish([0.,0.,0.],False)
        if len(self.results)>=self.episodes:self.stop('completed')
        else:self.mode='SELECT';self.sample_attempts=0

    def choose_goal(self):
        base,_=self.pose()
        origin=self.map.info.origin;q=origin.orientation
        origin_pose=[origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])]
        values=np.asarray(self.map.data,dtype=np.int16).reshape(self.map.info.height,self.map.info.width)
        candidates=reachable_goal_candidates(values,self.map.info.resolution,origin_pose,base,
            clearance=.28,min_distance=.55,max_distance=1.75)
        if len(candidates)==0:raise RuntimeError('No observed footprint-cleared goal candidates')
        return candidates[self.rng.integers(len(candidates))]

    def request_plan(self,goal,replan=False):
        request=ComputePathToPose.Goal()
        request.goal.header.frame_id='map';request.goal.header.stamp=self.get_clock().now().to_msg()
        request.goal.pose.position.x=float(goal[0]);request.goal.pose.position.y=float(goal[1]);request.goal.pose.orientation.w=1.
        request.planner_id='GridBased';request.use_start=False
        self.plan_deadline=time.monotonic()+5.;self.plan_pending=True
        self.plan_generation+=1;generation=self.plan_generation
        session=self.session_id
        future=self.planner.send_goal_async(request)
        def accepted(future):
            if not self.active or session!=self.session_id or generation!=self.plan_generation:return
            handle=future.result()
            if handle is None or not handle.accepted:
                self.plan_failed('rejected',replan);return
            handle.get_result_async().add_done_callback(completed)
        def completed(future):
            if not self.active or session!=self.session_id or generation!=self.plan_generation:return
            response=future.result();self.plan_pending=False
            if response is None or response.status!=4 or len(response.result.path.poses)<2:
                self.plan_failed('no_path',replan);return
            self.path=np.asarray([[p.pose.position.x,p.pose.position.y] for p in response.result.path.poses])
            self.last_replan=time.monotonic()
            if not replan:
                self.goal=goal.copy();self.goal_started=time.monotonic();self.mode='TRACK'
                self.event('goal_started',index=len(self.results)+1,goal=goal.tolist(),path=self.path.tolist())
        future.add_done_callback(accepted)

    def plan_failed(self,reason,replan):
        self.plan_pending=False
        if replan:
            self.last_replan=time.monotonic();self.event('replan_failed',reason=reason);return
        self.rejected_goals.append({'reason':reason,'sample_attempt':self.sample_attempts})
        self.mode='SELECT'

    def tick(self):
        try:self.update()
        except Exception as exc:
            self.event('controller_exception',error=str(exc))
            if self.active and self.goal is not None:
                try:self.finish_goal('fault',reason=str(exc))
                except Exception:pass
            self.stop('controller_exception:'+str(exc))

    def update(self):
        now=time.monotonic()
        if not self.active:
            if not self.commissioning_paused:self.publish([0.,0.,0.],False)
            return
        health=self.health();self.last_health=health
        if health:
            if self.goal is not None:self.finish_goal('fault',reason=','.join(health))
            self.stop('sensor_or_localization_fault:'+','.join(health));return
        if self.plan_pending and now>self.plan_deadline:
            self.stop('planner_timeout');return
        if self.mode=='SELECT':
            self.publish([0.,0.,0.],False)
            self.sample_attempts+=1
            if self.sample_attempts>100:self.stop('no_reachable_goals');return
            goal=self.choose_goal();self.mode='PLAN';self.request_plan(goal);return
        if self.mode=='PLAN':self.publish([0.,0.,0.],False);return
        base,_=self.pose();distance=float(np.linalg.norm(self.goal-base[:2]))
        measured=self.measured_odom.pose.pose;p=measured.position;q=measured.orientation
        fault=self.motion_progress.update(now,[p.x,p.y,yaw_of([q.x,q.y,q.z,q.w])],self.guard.get('applied',[0.,0.,0.]))
        if fault:
            self.finish_goal('fault',reason=fault);self.stop(fault);return
        if distance<=.15:
            self.publish([0.,0.,0.],True)
            v=self.odom.twist.twist
            if np.hypot(v.linear.x,v.linear.y)<.03 and abs(v.angular.z)<.1:
                if self.settled_at is None:self.settled_at=now
                elif now-self.settled_at>=.7:self.finish_goal('success')
            else:self.settled_at=None
            return
        self.settled_at=None
        if now-self.goal_started>self.goal_seconds:self.finish_goal('timeout');return
        target=subgoal_on_path(self.path,base,lookahead=.5)
        state,grid,details=self.observations(target,self.previous_action)
        inference_started=time.perf_counter()
        with torch.no_grad():action=self.actor(torch.from_numpy(state),torch.from_numpy(grid)).numpy()[0]
        if time.perf_counter()-inference_started>.1:
            self.stop('inference_deadline');return
        if self.health():
            self.stop('post_inference_health_fault');return
        if not np.isfinite(action).all():raise RuntimeError('Non-finite policy output')
        action=np.clip(action,-1.,1.)
        desired=np.clip(action*np.array([.2,.2,.5]),[-.1,-.1,-.3],[.1,.1,.3])
        velocity=self.last_velocity+np.clip(desired-self.last_velocity,[-.01,-.01,-.025],[.01,.01,.025])
        self.publish(velocity,True);self.last_velocity=velocity;self.previous_action=action
        # Persistent collision filtering is a stop event, not a silent success.
        stopped_by_guard='obstacle_close' in self.guard.get('blocked_by',[])
        stopped_by_filter=np.linalg.norm(velocity)>.025 and np.linalg.norm(self.filtered_velocity)<1e-5
        if stopped_by_guard or stopped_by_filter:
            if self.intervention_started is None:self.intervention_started=now
            elif now-self.intervention_started>.5:
                self.finish_goal('fault',reason='collision_stop_intervention');self.stop('collision_stop_intervention');return
        else:self.intervention_started=None
        if not self.plan_pending and now-self.last_replan>2.:self.request_plan(self.goal,replan=True)
        if now-self.last_status>.5:
            self.event('control',pose=base.tolist(),goal_distance=distance,subgoal=target.tolist(),
                policy_action=action.tolist(),requested_velocity=velocity.tolist(),guard=self.guard,
                min_scan_clearance=details['min_scan_clearance'],free_cells=details['free_cells'])
            status=String();status.data=json.dumps({'mode':self.mode,'completed_goals':len(self.results),'goal':self.goal.tolist(),'distance':distance})
            self.status_pub.publish(status);self.last_status=now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goals',type=int,default=30);parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--goal-seconds',type=float,default=90.)
    args=parser.parse_args()
    if not 1<=args.goals<=100 or args.goal_seconds<=0:raise ValueError('Invalid session limits')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);rclpy.init();node=AutonomousTrials(args.goals,args.seed,args.goal_seconds)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        try:node.stop('process_shutdown')
        except Exception:pass
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
