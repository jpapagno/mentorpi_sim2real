"""Live ROS sensor/TF adapter. This module never publishes movement commands."""
import math
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from sensor_msgs.msg import LaserScan,Imu
from nav_msgs.msg import OccupancyGrid,Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer,TransformListener
from navigation_math import yaw_of,rotation_of,occupancy_from_map,overlay_points,state_vector,map_alignment_score
from live_occupancy import live_occupancy_from_scan


class NavigationIO(Node):
    def __init__(self,name='mentorpi_navigation_io',local_mode=None):
        super().__init__(name)
        self.map=None;self.scan=None;self.odom=None
        self.measured_odom=None
        self.amcl_pose=None
        mode_file=Path('/home/ubuntu/shared/mentorpi_nav/config/localization_mode.json')
        mode=json.loads(mode_file.read_text()).get('mode') if mode_file.exists() else 'amcl'
        self.local_mode=(mode=='local') if local_mode is None else local_mode
        self.pose_frame='odom' if self.local_mode else 'map'
        self.require_amcl=not self.local_mode and mode=='amcl'
        self.map_match_checked=float('-inf')
        self.map_match=None
        self.received={}
        self.buffer=Buffer()
        self.listener=TransformListener(self.buffer,self)
        map_qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        if not self.local_mode:
            self.create_subscription(OccupancyGrid,'/map',lambda m:self.remember('map',m),map_qos)
            self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',lambda m:self.remember('amcl_pose',m),map_qos)
        for topic,key,kind in [('/scan','scan',LaserScan),('/imu','imu',Imu),('/odom','odom',Odometry),
            ('/mentorpi/measured_odom','measured_odom',Odometry)]:
            self.create_subscription(kind,topic,lambda m,k=key:self.remember(k,m),qos_profile_sensor_data)

    def remember(self,key,msg):
        setattr(self,key,msg)
        self.received[key]=time.monotonic()

    def age(self,msg):
        return self.get_clock().now().nanoseconds*1e-9-(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)

    def transform(self,target,source):
        transform=self.buffer.lookup_transform(target,source,rclpy.time.Time())
        t,q=transform.transform.translation,transform.transform.rotation
        return np.array([t.x,t.y,t.z]),rotation_of([q.x,q.y,q.z,q.w]),transform

    def pose(self):
        t,rotation,transform=self.transform(self.pose_frame,'base_footprint')
        return np.array([t[0],t[1],math.atan2(rotation[1,0],rotation[0,0])]),transform

    def health(self):
        reasons=[]
        for key,limit in [('map',float('inf')),('scan',.4),('imu',.3),('odom',.3),('measured_odom',.4)]:
            if key=='map' and self.local_mode:continue
            msg=getattr(self,key,None)
            if msg is None or time.monotonic()-self.received.get(key,float('-inf'))>limit:
                reasons.append(key+'_missing_or_stale')
            elif key!='map' and not -.05<=self.age(msg)<limit:
                reasons.append(key+'_stamp_stale')
        try:
            _,transform=self.pose()
            stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1e-9
            if self.get_clock().now().nanoseconds*1e-9-stamp>.4:reasons.append('tf_stale')
        except Exception:reasons.append('tf_missing')
        if self.scan is not None:
            try:self.transform('base_footprint',self.scan.header.frame_id)
            except Exception:reasons.append('lidar_tf_missing')
        if self.require_amcl:
            if self.amcl_pose is None:reasons.append('localization_missing')
            else:
                covariance=self.amcl_pose.pose.covariance
                if max(covariance[0],covariance[7])>.01 or covariance[35]>.05:
                    reasons.append('localization_uncertain')
            if self.map is not None and self.scan is not None and self.amcl_pose is not None:
                try:
                    if time.monotonic()-self.map_match_checked>.5:
                        base,_=self.pose();origin=self.map.info.origin;q=origin.orientation
                        values=np.asarray(self.map.data,dtype=np.int16).reshape(self.map.info.height,self.map.info.width)
                        self.map_match=map_alignment_score(values,self.map.info.resolution,
                            [origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])],base,self.scan_points())
                        self.map_match_checked=time.monotonic()
                    if self.map_match['matched_fraction']<.6:reasons.append('localization_scan_mismatch')
                except Exception:reasons.append('localization_consistency_unavailable')
        return reasons

    def scan_points(self):
        msg=self.scan
        ranges=np.asarray(msg.ranges)
        angle=msg.angle_min+np.arange(len(ranges))*msg.angle_increment
        valid=np.isfinite(ranges)&(ranges>=msg.range_min)&(ranges<=msg.range_max)
        points=np.column_stack((ranges[valid]*np.cos(angle[valid]),ranges[valid]*np.sin(angle[valid]),np.zeros(valid.sum())))
        t,rotation,_=self.transform('base_footprint',msg.header.frame_id)
        return points@rotation.T+t

    def observations(self,goal,previous_action):
        reasons=self.health()
        if reasons:raise RuntimeError(','.join(reasons))
        base,_=self.pose()
        scan_points=self.scan_points()
        if self.local_mode:
            sensor_origin,_,_=self.transform('base_footprint',self.scan.header.frame_id)
            blocked=live_occupancy_from_scan(scan_points,sensor_origin[:2])
        else:
            origin=self.map.info.origin;q=origin.orientation
            origin_pose=[origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])]
            values=np.asarray(self.map.data,dtype=np.int16).reshape(self.map.info.height,self.map.info.width)
            blocked=occupancy_from_map(values,self.map.info.resolution,origin_pose,base)
            blocked=overlay_points(blocked,scan_points[:,:2])
        v=self.measured_odom.twist.twist
        state=state_vector(goal,base,[v.linear.x,v.linear.y,v.angular.z],previous_action)
        return state[None,:],blocked.astype(np.float32)[None,None,:,:],{
            'pose':base.tolist(),'scan_points':len(scan_points),
            'pose_frame':self.pose_frame,'observation_source':'live_lidar' if self.local_mode else 'saved_map_plus_lidar',
            'free_cells':int((~blocked).sum()),'min_scan_clearance':float(np.linalg.norm(scan_points[:,:2],axis=1).min())}
