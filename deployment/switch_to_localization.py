"""Stop owned mapping and start the saved map/localizer; no movement commands."""
import json
import subprocess

stop_code="""
import os,signal,json
from pathlib import Path
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  owned_slam=args[0]==b'/opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node' and b'/home/ubuntu/shared/mentorpi_nav/config/slam.yaml' in args
  owned_perception=len(args)>1 and args[1]==b'/home/ubuntu/shared/mentorpi_nav/runtime/perception_node.py'
  if owned_slam or owned_perception:os.kill(int(path.parent.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
Path('/home/ubuntu/shared/mentorpi_nav/config/localization_mode.json').write_text(json.dumps({'mode':'amcl'}))
"""
subprocess.run(['docker','exec','MentorPi','python3','-c',stop_code],check=True)
commands={
 'map_server':"ros2 run nav2_map_server map_server --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/localization.yaml",
 'amcl':"ros2 run nav2_amcl amcl --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/localization.yaml",
 'localization_lifecycle':"ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=mentorpi_localization_lifecycle -p autostart:=true -p 'node_names:=[map_server,amcl]'",
 'perception':"python3 /home/ubuntu/shared/mentorpi_nav/runtime/perception_node.py",
}
for name,command in commands.items():
 command=f"source /home/ubuntu/.zshrc >/dev/null; exec {command} > /home/ubuntu/shared/mentorpi_nav/logs/{name}.log 2>&1"
 result=subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],capture_output=True,text=True)
 print(json.dumps({'component':name,'exit':result.returncode,'error':result.stderr}),flush=True)
