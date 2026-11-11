"""Read-only timing breakdown to distinguish stale sources from stale callbacks."""
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from navigation_io import NavigationIO

rclpy.init();node=NavigationIO('mentorpi_timing_probe')
rows=[]
start=time.monotonic()
while time.monotonic()-start<12:
    deadline=time.monotonic()+.05
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.005)
    row={'elapsed_s':time.monotonic()-start,'health':node.health()}
    now=node.get_clock().now().nanoseconds*1e-9
    for name in ('scan','imu','depth','odom'):
        msg=getattr(node,name,None)
        if msg is not None:
            row[name+'_stamp_age']=node.age(msg)
            row[name+'_receive_age']=time.monotonic()-node.received[name]
    for frame,target,source in [('map_base','map','base_footprint'),('map_odom','map','odom'),('odom_base','odom','base_footprint')]:
        try:
            transform=node.buffer.lookup_transform(target,source,rclpy.time.Time())
            stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1e-9
            row[frame+'_age']=now-stamp
        except Exception:pass
    rows.append(row)
summary={}
for key in sorted({key for row in rows for key in row if key.endswith('_age')}):
    values=[row[key] for row in rows if key in row]
    summary[key]={'count':len(values),'percentiles':np.percentile(values,[0,50,95,100]).tolist()}
faults={reason:sum(reason in row['health'] for row in rows) for reason in {reason for row in rows for reason in row['health']}}
result={'timings':summary,'health_fault_counts':faults,'rows':rows,'drive_commands_published':0}
Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/live_timing.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
