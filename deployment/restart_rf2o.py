import subprocess

stop_code="""
import os,signal
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if args[0].endswith(b'/rf2o_laser_odometry_node') and b'/home/ubuntu/shared/mentorpi_nav/config/rf2o.yaml' in args:
   os.kill(int(path.parent.name),signal.SIGTERM)
   print('Stopped owned RF2O PID',path.parent.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_code],check=True)
command="source /home/ubuntu/.zshrc >/dev/null; exec ros2 run rf2o_laser_odometry rf2o_laser_odometry_node --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/rf2o.yaml > /home/ubuntu/shared/mentorpi_nav/logs/rf2o.log 2>&1"
subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],check=True)
