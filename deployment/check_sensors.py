"""Fast live regression check for missing navigation sensor samples."""
import json
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Imu

rclpy.init()
node=rclpy.create_node('mentorpi_sensor_healthcheck')
counts={'/scan_raw':0,'/imu':0,'/ros_robot_controller/imu_raw':0}
subscriptions=[]
for topic in counts:
    def receive(msg,t=topic): counts[t]+=1
    subscriptions.append(node.create_subscription(LaserScan if topic=='/scan_raw' else Imu,
        topic,receive,qos_profile_sensor_data))
deadline=time.monotonic()+5
while time.monotonic()<deadline:
    rclpy.spin_once(node,timeout_sec=.1)
passed=all(count>=3 for count in counts.values())
print(json.dumps({'passed':passed,'sample_counts':counts}),flush=True)
node.destroy_node()
rclpy.shutdown()
raise SystemExit(0 if passed else 1)
