"""Live ROS sensor/TF adapter. This module never publishes movement commands."""
import math
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from sensor_msgs.msg import LaserScan,Imu,Image,CameraInfo
from nav_msgs.msg import OccupancyGrid,Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer,TransformListener
from navigation_math import yaw_of,rotation_of,occupancy_from_map,overlay_points,state_vector,map_alignment_score


class NavigationIO(Node):
    def __init__(self,name='mentorpi_navigation_io'):
        super().__init__(name)
        self.map=None;self.scan=None;self.depth=None;self.intrinsics=None;self.odom=None
        self.amcl_pose=None
        mode_file=Path('/home/ubuntu/shared/mentorpi_nav/config/localization_mode.json')
        self.require_amcl=mode_file.exists() and json.loads(mode_file.read_text()).get('mode')=='amcl'
        self.map_match_checked=float('-inf')
        self.map_match=None
        self.received={}
        self.buffer=Buffer()
        self.listener=TransformListener(self.buffer,self)
        map_qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid,'/map',lambda m:self.remember('map',m),map_qos)
        self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',lambda m:self.remember('amcl_pose',m),map_qos)
        for topic,key,kind in [('/scan','scan',LaserScan),('/imu','imu',Imu),('/odom','odom',Odometry),
            ('/ascamera/camera_publisher/depth0/image_raw','depth',Image),
            ('/ascamera/camera_publisher/depth0/camera_info','intrinsics',CameraInfo)]:
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT) if kind in (Image,CameraInfo) else qos_profile_sensor_data
            self.create_subscription(kind,topic,lambda m,k=key:self.remember(k,m),qos)

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
        t,rotation,transform=self.transform('map','base_footprint')
        return np.array([t[0],t[1],math.atan2(rotation[1,0],rotation[0,0])]),transform

    def health(self):
        reasons=[]
        for key,limit in [('map',float('inf')),('scan',.4),('imu',.3),('depth',.6),('odom',.3),('intrinsics',float('inf'))]:
            msg=getattr(self,key,None)
            if msg is None or time.monotonic()-self.received.get(key,float('-inf'))>limit:
                reasons.append(key+'_missing_or_stale')
            elif key not in ('map','intrinsics') and not -.05<=self.age(msg)<limit:
                reasons.append(key+'_stamp_stale')
        try:
            _,transform=self.pose()
            stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1e-9
            if self.get_clock().now().nanoseconds*1e-9-stamp>.4:reasons.append('tf_stale')
        except Exception:reasons.append('tf_missing')
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

    def depth_points(self):
        msg=self.depth
        if msg.encoding not in ('16UC1','32FC1'):raise ValueError('Unsupported depth encoding')
        dtype=np.uint16 if msg.encoding=='16UC1' else np.float32
        depth=np.frombuffer(msg.data,dtype=dtype).reshape(msg.height,-1)[:,:msg.width][::8,::8].astype(np.float32)
        if dtype==np.uint16:depth*=.001
        vv,uu=np.meshgrid(np.arange(0,msg.height,8),np.arange(0,msg.width,8),indexing='ij')
        fx,fy,cx,cy=self.intrinsics.k[0],self.intrinsics.k[4],self.intrinsics.k[2],self.intrinsics.k[5]
        valid=np.isfinite(depth)&(depth>.18)&(depth<4.)
        z=depth[valid]
        points=np.column_stack(((uu[valid]-cx)*z/fx,(vv[valid]-cy)*z/fy,z))
        t,rotation,_=self.transform('base_footprint',msg.header.frame_id)
        points=points@rotation.T+t
        radius=np.linalg.norm(points[:,:2],axis=1)
        return points[(points[:,2]>.04)&(points[:,2]<.5)&(radius>.17)]

    def observations(self,goal,previous_action):
        reasons=self.health()
        if reasons:raise RuntimeError(','.join(reasons))
        base,_=self.pose()
        origin=self.map.info.origin
        q=origin.orientation
        origin_pose=[origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])]
        values=np.asarray(self.map.data,dtype=np.int16).reshape(self.map.info.height,self.map.info.width)
        blocked=occupancy_from_map(values,self.map.info.resolution,origin_pose,base)
        scan_points=self.scan_points()
        depth_points=self.depth_points()
        blocked=overlay_points(blocked,scan_points[:,:2])
        blocked=overlay_points(blocked,depth_points[:,:2])
        v=self.odom.twist.twist
        state=state_vector(goal,base,[v.linear.x,v.linear.y,v.angular.z],previous_action)
        return state[None,:],blocked.astype(np.float32)[None,None,:,:],{
            'pose':base.tolist(),'scan_points':len(scan_points),'depth_points':len(depth_points),
            'free_cells':int((~blocked).sum()),'min_scan_clearance':float(np.linalg.norm(scan_points[:,:2],axis=1).min())}
