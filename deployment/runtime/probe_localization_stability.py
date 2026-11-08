"""Measure stationary pose drift against zero hardware output; no enabled motion."""
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from navigation_io import NavigationIO
from navigation_math import yaw_of

rclpy.init();node=NavigationIO('mentorpi_localization_stability')
rows={'odom':[],'rf2o':[],'map':[],'imu_yaw':[],'imu_raw_gyro':[]}
enable_pub=node.create_publisher(Bool,'/mentorpi/motion_enable',10)

def record_odom(msg,key):
    p,q=msg.pose.pose.position,msg.pose.pose.orientation
    rows[key].append([p.x,p.y,yaw_of([q.x,q.y,q.z,q.w])])

def record_imu(msg):
    q=msg.orientation;rows['imu_yaw'].append([yaw_of([q.x,q.y,q.z,q.w])])

def record_raw(msg):rows['imu_raw_gyro'].append([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z])

node.create_subscription(Odometry,'/odom',lambda m:record_odom(m,'odom'),10)
node.create_subscription(Odometry,'/odom_rf2o',lambda m:record_odom(m,'rf2o'),10)
node.create_subscription(Imu,'/imu',record_imu,10)
node.create_subscription(Imu,'/ros_robot_controller/imu_raw',record_raw,10)
started=time.monotonic();next_sample=started
while time.monotonic()-started<12:
    rclpy.spin_once(node,timeout_sec=.005)
    if time.monotonic()>=next_sample:
        message=Bool();message.data=False;enable_pub.publish(message)
        try:rows['map'].append(node.pose()[0].tolist())
        except Exception:pass
        next_sample=time.monotonic()+.1
result={}
for key,values in rows.items():
    if not values:result[key]={'samples':0};continue
    array=np.asarray(values)
    result[key]={'samples':len(array),'first':array[0].tolist(),'last':array[-1].tolist(),
                 'span':np.ptp(array,axis=0).tolist(),'mean':array.mean(axis=0).tolist()}
map_span=result['map'].get('span',[float('inf')]*3)
result['passed']=bool(np.linalg.norm(map_span[:2])<.03 and map_span[2]<math.radians(3))
Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/localization_stability.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
raise SystemExit(0 if result['passed'] else 1)
