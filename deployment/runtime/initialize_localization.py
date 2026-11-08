"""Initialize the saved-map pose automatically from the mapping handoff."""
import json
import math
from pathlib import Path
import time
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from navigation_io import NavigationIO

rclpy.init();node=NavigationIO('mentorpi_initialize_localization')
publisher=node.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)
pose=json.loads(Path('/home/ubuntu/shared/mentorpi_nav/config/initial_pose.json').read_text())
deadline=time.monotonic()+15;sent=0;next_send=time.monotonic()+3
while time.monotonic()<deadline:
 rclpy.spin_once(node,timeout_sec=.02)
 if sent<3 and time.monotonic()>next_send and node.map is not None and node.count_subscribers('/initialpose'):
  msg=PoseWithCovarianceStamped();msg.header.frame_id='map';msg.header.stamp=node.get_clock().now().to_msg()
  msg.pose.pose.position.x=pose[0];msg.pose.pose.position.y=pose[1]
  msg.pose.pose.orientation.z=math.sin(pose[2]/2);msg.pose.pose.orientation.w=math.cos(pose[2]/2)
  msg.pose.covariance[0]=.0025;msg.pose.covariance[7]=.0025;msg.pose.covariance[35]=.01
  publisher.publish(msg);sent+=1;next_send=time.monotonic()+1
result={'initializations_sent':sent,'health':node.health()}
if node.amcl_pose is not None:result['covariance']=list(node.amcl_pose.pose.covariance)
print(json.dumps(result,indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
raise SystemExit(0 if sent and not result['health'] else 1)
