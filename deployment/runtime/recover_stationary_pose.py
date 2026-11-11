"""Read-only scan fitting by default; --apply resets localization, never motors."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import uuid
import yaml
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
import numpy as np
from navigation_io import NavigationIO
from navigation_math import yaw_of
from scan_relocalization import fit_scan_pose

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--apply',action='store_true')
args=parser.parse_args()
root=Path('/home/ubuntu/shared/mentorpi_nav')
rclpy.init();node=NavigationIO('mentorpi_stationary_pose_recovery')
guards=[]
node.create_subscription(String,'/mentorpi/guard_status',lambda m:guards.append(json.loads(m.data)),10)
deadline=time.monotonic()+8
while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
if node.map is None or node.scan is None or node.age(node.scan)>.4:raise RuntimeError('Fresh map/scan required')
if not guards or 'disarmed' not in guards[-1].get('blocked_by',[]) or any(any(abs(v)>1e-9 for v in g['applied']) for g in guards):
    raise RuntimeError('Robot must remain disarmed and stationary')
base,_=node.pose();m=node.map;o=m.info.origin;q=o.orientation
values=np.asarray(m.data).reshape(m.info.height,m.info.width)
result=fit_scan_pose(values,m.info.resolution,[o.position.x,o.position.y,yaw_of([q.x,q.y,q.z,q.w])],node.scan_points(),base)
result.update(applied=False,drive_commands_published=0)
if args.apply:
    if result['after']['matched_fraction']<.85 or result['after']['median_error_m']>.1:
        raise RuntimeError('Scan registration not strong enough to apply: '+str(result))
    pose=result['pose']
    publisher=node.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)
    for _ in range(3):
        msg=PoseWithCovarianceStamped();msg.header.frame_id='map';msg.header.stamp=node.get_clock().now().to_msg()
        msg.pose.pose.position.x=pose[0];msg.pose.pose.position.y=pose[1]
        msg.pose.pose.orientation.z=math.sin(pose[2]/2);msg.pose.pose.orientation.w=math.cos(pose[2]/2)
        msg.pose.covariance[0]=.0025;msg.pose.covariance[7]=.0025;msg.pose.covariance[35]=.01
        publisher.publish(msg)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
    node.map_match_checked=float('-inf')
    result.update(applied=True,health=node.health(),verified_alignment=node.map_match)
    if result['health']:raise RuntimeError('Localization still unhealthy: '+str(result))
    actual,_=node.pose()
    backup=root/'diagnostics'/('pose_before_recovery_'+uuid.uuid4().hex[:8]);backup.mkdir()
    for name in ('initial_pose.json','localization.yaml'):
        shutil.copy2(root/'config'/name,backup/name)
    (root/'config/initial_pose.json').write_text(json.dumps(actual.tolist()))
    config=yaml.safe_load((root/'config/localization.yaml').read_text())
    config['amcl']['ros__parameters']['initial_pose'].update(x=float(actual[0]),y=float(actual[1]),yaw=float(actual[2]))
    (root/'config/localization.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    result['saved_pose']=actual.tolist()
(root/'diagnostics/stationary_pose_recovery.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
