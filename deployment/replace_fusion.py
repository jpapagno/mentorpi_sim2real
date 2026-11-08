"""Replace conflicting vendor heading fusion while the hardware is disarmed."""
import subprocess

stop_code="""
import os,signal
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  own_slam=args[0]==b'/opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node' and b'/home/ubuntu/shared/mentorpi_nav/config/slam.yaml' in args
  old_ekf=args[0]==b'/opt/ros/humble/lib/robot_localization/ekf_node' and b'/home/ubuntu/shared/mentorpi_nav/config/ekf.yaml' not in args
  if own_slam or old_ekf:
   os.kill(int(path.parent.name),signal.SIGTERM)
   print('Stopped verified fusion/mapping PID',path.parent.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_code],check=True)
commands=[
 "source /home/ubuntu/.zshrc >/dev/null; exec ros2 run robot_localization ekf_node --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/ekf.yaml -r odometry/filtered:=odom > /home/ubuntu/shared/mentorpi_nav/logs/ekf.log 2>&1",
 "source /home/ubuntu/.zshrc >/dev/null; exec ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/slam.yaml > /home/ubuntu/shared/mentorpi_nav/logs/slam.log 2>&1",
]
for command in commands:
 subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],check=True)
