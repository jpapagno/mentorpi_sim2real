"""Switch from inspected prototype/vendor bringup to the complete disarmed unit."""
import subprocess
import time

subprocess.run(['sudo','-n','systemctl','disable','--now','start_node.service'],check=True)
stop_code="""
import os,signal
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  vendor=b'launch' in args and b'bringup' in args and b'bringup.launch.py' in args
  owned=any(arg.startswith(b'/home/ubuntu/shared/mentorpi_nav/runtime/') or arg.startswith(b'/home/ubuntu/shared/mentorpi_nav/config/') for arg in args)
  if vendor or owned:
   os.kill(int(path.parent.name),signal.SIGINT)
   print('Stopped previous component PID',path.parent.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_code],check=True)
time.sleep(5)
# Stop the temporary LiDAR started before persistent configuration existed.
stop_lidar="""
import os,signal
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if args[0].endswith(b'/ldlidar_stl_ros2_node') and b'/tmp/launch_params_v9xkjvj3' in args:os.kill(int(path.parent.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_lidar],check=True)
subprocess.run(['sudo','-n','install','-m','644','/home/pi/docker/tmp/mentorpi_nav/mentorpi-navigation.service','/etc/systemd/system/mentorpi-navigation.service'],check=True)
subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
subprocess.run(['sudo','-n','systemctl','enable','--now','mentorpi-navigation.service'],check=True)
print('Persistent stack enabled; its trial controller starts DISARMED',flush=True)
