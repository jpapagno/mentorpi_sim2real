"""Save the observed map and generate fixed-map AMCL configuration."""
import json
import argparse
from pathlib import Path
import subprocess
import time
import yaml
import rclpy
from navigation_io import NavigationIO

root=Path('/home/ubuntu/shared/mentorpi_nav')
parser=argparse.ArgumentParser()
parser.add_argument('--map-name',default='test_area')
args=parser.parse_args()
if not args.map_name.replace('_','').isalnum():raise ValueError('Simple map name required')
map_path=root/'maps'/args.map_name
(root/'maps').mkdir(exist_ok=True)
rclpy.init();node=NavigationIO('mentorpi_save_test_area')
deadline=time.monotonic()+8
while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
if node.health():raise RuntimeError(str(node.health()))
pose,_=node.pose()
command=['ros2','run','nav2_map_server','map_saver_cli','-f',str(map_path),
         '--ros-args','-p','save_map_timeout:=20.0','-p','map_subscribe_transient_local:=true']
run=subprocess.run(command,capture_output=True,text=True,timeout=30)
if run.returncode:raise RuntimeError(run.stdout+run.stderr)
parameters={
 'use_sim_time':False,'global_frame_id':'map','odom_frame_id':'odom','base_frame_id':'base_footprint',
 'scan_topic':'/scan','tf_broadcast':True,'transform_tolerance':.3,
 'robot_model_type':'nav2_amcl::OmniMotionModel','alpha1':.1,'alpha2':.1,'alpha3':.2,'alpha4':.2,'alpha5':.2,
 'laser_model_type':'likelihood_field','laser_min_range':.1,'laser_max_range':8.,
 'max_beams':90,'min_particles':200,'max_particles':1000,
 'z_hit':.9,'z_rand':.1,'sigma_hit':.15,'laser_likelihood_max_dist':1.5,
 'update_min_d':.025,'update_min_a':.025,'resample_interval':1,
 'set_initial_pose':True,'always_reset_initial_pose':False,
 'initial_pose':{'x':float(pose[0]),'y':float(pose[1]),'z':0.,'yaw':float(pose[2])},
}
configuration={'amcl':{'ros__parameters':parameters},
 'map_server':{'ros__parameters':{'use_sim_time':False,'yaml_filename':str(map_path.with_suffix('.yaml')),'frame_id':'map','topic_name':'map'}}}
(root/'config/localization.yaml').write_text(yaml.safe_dump(configuration,sort_keys=False))
(root/'config/initial_pose.json').write_text(json.dumps(pose.tolist()))
print(json.dumps({'saved_map':str(map_path.with_suffix('.yaml')),'pose':pose.tolist(),'map_width':node.map.info.width,'map_height':node.map.info.height}),flush=True)
node.destroy_node();rclpy.shutdown()
