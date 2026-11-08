"""Read active OS processes and the installed IMU launch source."""
import json
import subprocess

code="""
import json
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
processes=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if args[0].endswith(b'/rf2o_laser_odometry_node') or args[0].endswith(b'/lifecycle_manager'):
   processes.append({'pid':int(path.parent.name),'args':[a.decode(errors='replace') for a in args if a]})
 except (FileNotFoundError,PermissionError):pass
source=Path(get_package_share_directory('peripherals'))/'launch/imu_filter.launch.py'
print(json.dumps({'processes':processes,'imu_launch_file':str(source),'imu_launch':source.read_text()},indent=2))
"""
command="source /home/ubuntu/.zshrc >/dev/null; python3 -"
run=subprocess.run(['docker','exec','-i','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],input=code,text=True,capture_output=True,timeout=20)
print(run.stdout,flush=True)
print(run.stderr,flush=True)
