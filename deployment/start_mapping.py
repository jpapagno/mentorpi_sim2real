"""Start stationary mapping and Nav2 planning; neither publishes drive commands."""
import json
import subprocess

commands={
 'mapping':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/slam.yaml > /home/ubuntu/shared/mentorpi_nav/logs/slam.log 2>&1",
 'planner':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run nav2_planner planner_server --ros-args --params-file /home/ubuntu/shared/mentorpi_nav/config/planner.yaml > /home/ubuntu/shared/mentorpi_nav/logs/planner.log 2>&1",
 'planner_lifecycle':"source /home/ubuntu/.zshrc >/dev/null; exec ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=mentorpi_planner_lifecycle -p autostart:=true -p 'node_names:=[planner_server]' > /home/ubuntu/shared/mentorpi_nav/logs/planner_lifecycle.log 2>&1",
}
result={}
for name,command in commands.items():
 r=subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],capture_output=True,text=True)
 result[name]={'exit':r.returncode,'error':r.stderr}
print(json.dumps(result,indent=2),flush=True)
