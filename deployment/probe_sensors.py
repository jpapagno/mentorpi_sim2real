"""Stationary measurement of live sensors and TF. Never publishes commands."""
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener


class Probe(Node):
    def __init__(self):
        super().__init__('mentorpi_stationary_probe')
        self.samples = {}
        self.last = {}
        self.subscriptions_live = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        for topic, kind in {
            '/scan_raw':LaserScan, '/odom':Odometry, '/odom_raw':Odometry,
            '/odom_rf2o':Odometry, '/imu':Imu,
            '/ascamera/camera_publisher/rgb0/image':Image,
            '/ascamera/camera_publisher/depth0/image_raw':Image,
            '/ascamera/camera_publisher/depth0/points':PointCloud2,
            '/controller/cmd_vel':Twist, '/cmd_vel':Twist,
        }.items():
            self.samples[topic] = []
            self.subscriptions_live.append(self.create_subscription(kind,topic,
                lambda msg,t=topic:self.receive(t,msg),qos_profile_sensor_data))

    def receive(self,topic,msg):
        self.samples[topic].append(time.monotonic())
        self.last[topic] = msg


def transform_dict(transform):
    t,q = transform.transform.translation,transform.transform.rotation
    return {'parent':transform.header.frame_id,'child':transform.child_frame_id,
            'translation':[t.x,t.y,t.z],'quaternion':[q.x,q.y,q.z,q.w]}


def main():
    rclpy.init()
    node = Probe()
    started = time.monotonic()
    while time.monotonic()-started < 15:
        rclpy.spin_once(node,timeout_sec=.1)
    result = {'duration_s':time.monotonic()-started,'topics':{},'transforms':{}}
    output = Path('/home/ubuntu/shared/mentorpi_nav/diagnostics')
    output.mkdir(parents=True,exist_ok=True)
    for topic, arrivals in node.samples.items():
        info={'count':len(arrivals),'hz':(len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else 0}
        msg=node.last.get(topic)
        if msg is None:
            result['topics'][topic]=info
            continue
        if hasattr(msg,'header'):
            info.update(frame=msg.header.frame_id,stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                        age_s=node.get_clock().now().nanoseconds*1e-9-(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9))
        if isinstance(msg,LaserScan):
            values=np.asarray(msg.ranges)
            finite=values[np.isfinite(values)&(values>=msg.range_min)&(values<=msg.range_max)]
            info.update(points=len(values),valid=len(finite),range_min=msg.range_min,range_max=msg.range_max,
                        angle_min=msg.angle_min,angle_increment=msg.angle_increment,
                        nearest_m=float(finite.min()) if len(finite) else None,
                        range_percentiles=np.percentile(finite,[0,10,50,90,100]).tolist() if len(finite) else [])
            np.savez_compressed(output/'scan.npz',ranges=values,angle_min=msg.angle_min,angle_increment=msg.angle_increment)
        elif isinstance(msg,Odometry):
            p,q,v=msg.pose.pose.position,msg.pose.pose.orientation,msg.twist.twist
            info.update(child_frame=msg.child_frame_id,position=[p.x,p.y,p.z],quaternion=[q.x,q.y,q.z,q.w],
                        velocity=[v.linear.x,v.linear.y,v.angular.z],pose_covariance=list(msg.pose.covariance))
        elif isinstance(msg,Imu):
            q=msg.orientation
            info.update(quaternion=[q.x,q.y,q.z,q.w],angular_velocity=[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z])
        elif isinstance(msg,Image):
            info.update(encoding=msg.encoding,width=msg.width,height=msg.height,step=msg.step)
            if msg.encoding in ('rgb8','bgr8'):
                image=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
                if msg.encoding=='rgb8': image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output/'rgb.png'),image)
            elif msg.encoding in ('16UC1','mono16','32FC1'):
                dtype=np.uint16 if msg.encoding!='32FC1' else np.float32
                depth=np.frombuffer(msg.data,dtype=dtype).reshape(msg.height,-1)[:,:msg.width]
                scale=.001 if dtype==np.uint16 else 1.
                meters=depth.astype(np.float32)*scale
                valid=meters[np.isfinite(meters)&(meters>0)]
                info.update(valid_fraction=float(len(valid)/meters.size),depth_percentiles_m=np.percentile(valid,[0,10,50,90,100]).tolist() if len(valid) else [])
                np.save(output/'depth_m.npy',meters)
                preview=cv2.applyColorMap((np.clip(meters/4,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
                cv2.imwrite(str(output/'depth.png'),preview)
        elif isinstance(msg,PointCloud2):
            info.update(width=msg.width,height=msg.height,point_step=msg.point_step,
                        fields=[{'name':f.name,'offset':f.offset,'datatype':f.datatype} for f in msg.fields])
        elif isinstance(msg,Twist):
            info.update(command=[msg.linear.x,msg.linear.y,msg.angular.z])
        result['topics'][topic]=info
    frames={'odom','base_footprint','base_link','depth_cam'}
    frames.update(m.header.frame_id for m in node.last.values() if hasattr(m,'header'))
    for frame in sorted(frames):
        try:
            result['transforms'][frame]=transform_dict(node.tf_buffer.lookup_transform('base_footprint',frame,rclpy.time.Time()))
        except Exception as exc:
            result['transforms'][frame]={'error':str(exc)}
    (output/'sensors.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__': main()
