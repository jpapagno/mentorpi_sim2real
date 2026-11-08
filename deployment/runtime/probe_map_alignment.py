import json
import time
import numpy as np
import rclpy
from navigation_io import NavigationIO
from navigation_math import yaw_of,map_alignment_score

rclpy.init();node=NavigationIO('mentorpi_map_alignment_probe')
deadline=time.monotonic()+20
while time.monotonic()<deadline:
 rclpy.spin_once(node,timeout_sec=.02)
 if not node.health():break
health=node.health()
if health:raise RuntimeError(str(health))
base,_=node.pose();origin=node.map.info.origin;q=origin.orientation
values=np.asarray(node.map.data,dtype=np.int16).reshape(node.map.info.height,node.map.info.width)
score=map_alignment_score(values,node.map.info.resolution,
    [origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])],base,node.scan_points())
print(json.dumps({'pose':base.tolist(),'map_alignment':score,'drive_commands_published':0},indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
raise SystemExit(0 if score['matched_fraction']>=.6 else 1)
