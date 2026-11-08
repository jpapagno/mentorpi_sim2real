"""Replace only this project's SLAM process and add measured laser odometry."""
import json
import subprocess

stop_code="""
import os,signal
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if args[0]==b'/opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node' and b'/home/ubuntu/shared/mentorpi_nav/config/slam.yaml' in args:
   os.kill(int(path.parent.name),signal.SIGTERM)
   print('Stopped owned SLAM PID',path.parent.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_code],check=True)
commands={
 'scan_matched_odometry':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run rf2o_laser_odometry rf2o_laser_odometry_node --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/rf2o.yaml > /home/ubuntu/shared/mentorpi_nav/logs/rf2o.log 2>&1",
 'mapping':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/slam.yaml > /home/ubuntu/shared/mentorpi_nav/logs/slam.log 2>&1",
}
for name,command in commands.items():
 r=subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],capture_output=True,text=True)
 print(json.dumps({'component':name,'exit':r.returncode,'error':r.stderr}),flush=True)
