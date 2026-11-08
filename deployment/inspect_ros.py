"""Read-only ROS/container inventory, including selected configuration/source."""
import glob
import importlib.util
import json
from pathlib import Path
import os
import subprocess


def capture(command, timeout=20):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return {'exit': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except Exception as exc:
        return {'error': str(exc)}


commands = {
    'os': 'cat /etc/os-release; python3 --version',
    'nodes': 'ros2 node list',
    'topics': 'ros2 topic list -t',
    'actions': 'ros2 action list -t',
    'packages': "ros2 pkg list | grep -E 'navigation|nav2|slam|localization|controller|lidar|depth|pointcloud|ascamera|rplidar|rrc'",
    'model': "cat /proc/device-tree/model",
    'workspace': 'ls -ld /home/ubuntu/ros2_ws /home/ubuntu/shared /home/ubuntu/share /home/ubuntu/mentorpi* 2>/dev/null',
    'imu': 'ros2 topic info /imu -v',
    'odom': 'ros2 topic info /odom -v',
    'cmd_vel': 'ros2 topic info /controller/cmd_vel -v; ros2 topic info /cmd_vel -v',
    'scan': 'ros2 topic info /scan_raw -v; ros2 topic info /scan -v',
    'maps': "find /home/ubuntu -maxdepth 5 -type f \\( -name '*.pgm' -o -name '*.posegraph' -o -name '*map*.yaml' \\) 2>/dev/null | head -60",
}
patterns = [
    '/home/ubuntu/ros2_ws/src/driver/controller/controller/odom_publisher.py',
    '/home/ubuntu/ros2_ws/src/driver/controller/config/*.yaml',
    '/home/ubuntu/ros2_ws/src/driver/controller/launch/*.py',
    '/home/ubuntu/ros2_ws/src/navigation/**/launch/*.py',
    '/home/ubuntu/ros2_ws/src/navigation/**/config/*mecanum*.yaml',
    '/home/ubuntu/ros2_ws/src/slam/**/launch/*.py',
    '/home/ubuntu/ros2_ws/src/bringup/launch/*.py',
]
files = {}
for pattern in patterns:
    for filename in glob.glob(pattern, recursive=True):
        path = Path(filename)
        if path.is_file():
            files[filename] = path.read_text(errors='replace')[:40000]
print(json.dumps({
    'environment': {key:os.environ.get(key) for key in ['ROS_DISTRO','ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','RMW_IMPLEMENTATION','ROBOT_TYPE','DEPTH_CAMERA_TYPE','LIDAR_TYPE','PYTHONPATH']},
    'commands': {name:capture(command) for name,command in commands.items()},
    'python_modules': {name:importlib.util.find_spec(name) is not None for name in ['numpy','torch','onnxruntime','rclpy','cv2']},
    'files':files,
}, indent=2), flush=True)
