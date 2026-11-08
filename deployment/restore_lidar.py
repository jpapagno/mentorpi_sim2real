"""Replace the verified stale LiDAR process; only the scanning sensor is started."""
import json
import subprocess

# PID 3510 was verified inside the container with fd -> ttyUSB0 (deleted).
check=subprocess.run(['docker','exec','MentorPi','readlink','/proc/3510/fd/17'],capture_output=True,text=True)
if check.stdout.strip()!='/dev/ttyUSB0 (deleted)':
    raise RuntimeError('Driver identity changed; refusing to stop an unverified process')
subprocess.run(['docker','exec','MentorPi','kill','-TERM','3510'],check=True)
command="source /home/ubuntu/.zshrc >/dev/null; exec /home/ubuntu/third_party_ros2/third_party_ws/install/ldlidar_stl_ros2/lib/ldlidar_stl_ros2/ldlidar_stl_ros2_node --ros-args -r __node:=mentorpi_lidar --params-file /tmp/launch_params_v9xkjvj3 -r scan:=scan_raw > /home/ubuntu/shared/mentorpi_nav/logs/lidar.log 2>&1"
run=subprocess.run(['docker','exec','-d','-u','ubuntu','-w','/home/ubuntu','MentorPi','/bin/zsh','-c',command],capture_output=True,text=True)
print(json.dumps({'driver_restart_exit':run.returncode,'stderr':run.stderr}),flush=True)
raise SystemExit(run.returncode)
