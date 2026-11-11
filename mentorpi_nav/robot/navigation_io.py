"""Latest LiDAR and measured motion, converted to the actor's input tensors."""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from .navigation_math import rotation_of, state_vector
from .live_occupancy import live_occupancy_from_scan


class NavigationIO(Node):
    def __init__(self,name='mentorpi_navigation_io'):
        super().__init__(name)
        self.scan=self.imu=self.odom=self.measured_odom=None
        self.pose_frame='odom'
        self.received={}
        self.buffer=Buffer()
        self.listener=TransformListener(self.buffer,self)
        for topic,key,kind in [('/scan','scan',LaserScan),('/imu','imu',Imu),
                ('/odom','odom',Odometry),('/mentorpi/measured_odom','measured_odom',Odometry)]:
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
        t,rotation,transform=self.transform('odom','base_footprint')
        return np.array([t[0],t[1],math.atan2(rotation[1,0],rotation[0,0])]),transform

    def health(self):
        reasons=[]
        for key,limit in [('scan',.4),('imu',.3),('odom',.3),('measured_odom',.4)]:
            msg=getattr(self,key)
            if msg is None or time.monotonic()-self.received.get(key,float('-inf'))>limit:
                reasons.append(key+'_missing_or_stale')
            elif not -.05<=self.age(msg)<limit:reasons.append(key+'_stamp_stale')
        try:
            _,transform=self.pose()
            if self.age(transform)>.4:reasons.append('tf_stale')
        except Exception:reasons.append('tf_missing')
        if self.scan is not None:
            try:self.transform('base_footprint',self.scan.header.frame_id)
            except Exception:reasons.append('lidar_tf_missing')
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
        points=self.scan_points()
        origin,_,_=self.transform('base_footprint',self.scan.header.frame_id)
        blocked=live_occupancy_from_scan(points,origin[:2])
        v=self.measured_odom.twist.twist
        state=state_vector(goal,base,[v.linear.x,v.linear.y,v.angular.z],previous_action)
        return state[None,:],blocked.astype(np.float32)[None,None,:,:],{
            'pose':base.tolist(),'scan_points':len(points),'pose_frame':'odom',
            'observation_source':'live_lidar','free_cells':int((~blocked).sum()),
            'min_scan_clearance':float(np.linalg.norm(points[:,:2],axis=1).min())}
