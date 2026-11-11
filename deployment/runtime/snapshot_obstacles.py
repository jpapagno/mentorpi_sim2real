"""Read-only sensor/map evidence; renders current obstacle hits and policy grid."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from navigation_io import NavigationIO
from navigation_math import overlay_points

parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
args=parser.parse_args()
if not args.label.replace('_','').isalnum():raise ValueError('Use a simple snapshot label')
root=Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/obstacle_test')
root.mkdir(exist_ok=True)
rclpy.init();node=NavigationIO('mentorpi_obstacle_snapshot')
guards=[]
rgb=[]
node.create_subscription(Image,'/ascamera/camera_publisher/rgb0/image',lambda m:rgb.append(m) if not rgb else rgb.__setitem__(0,m),qos_profile_sensor_data)
node.create_subscription(String,'/mentorpi/guard_status',lambda m:guards.append(json.loads(m.data)),10)
started=time.monotonic()
while time.monotonic()-started<6:
    rclpy.spin_once(node,timeout_sec=.02)
scan=node.scan_points()
result={'label':args.label,'health':node.health(),'scan_points':len(scan),
        'guard_last':guards[-1] if guards else None,'drive_commands_published':0,
        'max_reported_motor_command':max((max(abs(v) for v in s['applied']) for s in guards),default=None)}
for name,points in [('lidar',scan)]:
    front=points[(points[:,0]>.2)&(points[:,0]<1.5)&(abs(points[:,1])<.35)]
    result[name+'_front_points']=len(front)
    result[name+'_front_nearest_m']=float(front[:,0].min()) if len(front) else None
    np.save(root/(args.label+'_'+name+'.npy'),points)

# Diagnostic hit plot. Background is unobserved, NOT certified free space.
hits=np.full((100,100,3),225,dtype=np.uint8)
lidar=overlay_points(np.zeros((100,100),bool),scan[:,:2],uncertainty_cells=0)
hits[lidar]=(210,90,30)
cv2.circle(hits,(50,50),3,(0,0,220),1)
image=cv2.resize(hits,(600,600),interpolation=cv2.INTER_NEAREST)
cv2.putText(image,'LIVE HITS: +x forward UP, +y left LEFT',(12,25),0,.55,(0,0,0),1)
cv2.putText(image,'Blue=LiDAR Gray=unclassified',(12,585),0,.52,(0,0,0),1)
cv2.imwrite(str(root/(args.label+'_hits.png')),image)
if rgb:
    msg=rgb[0]
    pixels=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
    if msg.encoding=='rgb8':pixels=cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(root/(args.label+'_rgb.png')),pixels)
if not result['health']:
    pose,_=node.pose()
    _,grid,info=node.observations(pose[:2],[0.,0.,0.])
    np.save(root/(args.label+'_policy_grid.npy'),grid[0,0])
    cv2.imwrite(str(root/(args.label+'_policy_grid.png')),cv2.resize(((1-grid[0,0])*255).astype(np.uint8),(600,600),interpolation=cv2.INTER_NEAREST))
    result['policy_grid_info']=info
if node.map is not None:
    m=node.map
    values=np.asarray(m.data).reshape(m.info.height,m.info.width)
    np.save(root/(args.label+'_map.npy'),values)
    gray=np.where(values<0,160,np.where(values>=50,0,255)).astype(np.uint8)
    cv2.imwrite(str(root/(args.label+'_map.png')),gray[::-1])
    result['map']={'width':m.info.width,'height':m.info.height,'resolution':m.info.resolution}
(root/(args.label+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
