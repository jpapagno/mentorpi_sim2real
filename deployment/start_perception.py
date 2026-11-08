import json
import subprocess

commands={
 'perception':"source /home/ubuntu/.zshrc >/dev/null; exec python3 /home/ubuntu/shared/mentorpi_nav/runtime/perception_node.py > /home/ubuntu/shared/mentorpi_nav/logs/perception.log 2>&1",
 'collision_monitor':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run nav2_collision_monitor collision_monitor --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/collision_monitor.yaml > /home/ubuntu/shared/mentorpi_nav/logs/collision_monitor.log 2>&1",
 'collision_lifecycle':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=mentorpi_collision_lifecycle -p autostart:=true -p 'node_names:=[collision_monitor]' > /home/ubuntu/shared/mentorpi_nav/logs/collision_lifecycle.log 2>&1",
}
for name,command in commands.items():
 result=subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],capture_output=True,text=True)
 print(json.dumps({'component':name,'exit':result.returncode,'error':result.stderr}),flush=True)
