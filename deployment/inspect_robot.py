"""Read-only host inventory; run through SSH with python3 on stdin."""
import importlib.util
import json
from pathlib import Path
import subprocess


def capture(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        return {'exit': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except Exception as exc:
        return {'error': str(exc)}


commands = {
    'identity': 'id; hostname; uname -a',
    'os': 'cat /etc/os-release',
    'resources': 'free -h; df -h / /dev/shm; uptime',
    'ros_installs': 'ls -la /opt/ros 2>/dev/null; ls -ld /home/pi/ros* /home/ubuntu/ros* 2>/dev/null',
    'containers': "docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}'",
    'services': "systemctl list-units --type=service --state=running --no-pager --no-legend",
    'robot_processes': "ps -eo pid,comm,args | grep -E 'ros2|launch|controller|lidar|ascamera|slam|nav2|ekf|docker' | grep -v grep",
    'workspaces': "find /home/pi /home/ubuntu -maxdepth 3 -type d \\( -name src -o -name maps -o -name navigation -o -name slam \\) 2>/dev/null | head -60",
    'hardware': 'ls -l /dev/serial/by-id /dev/video* /dev/lidar /dev/rrc 2>/dev/null; vcgencmd get_throttled 2>/dev/null',
}
print(json.dumps({
    'commands': {name: capture(command) for name, command in commands.items()},
    'python_modules': {name: importlib.util.find_spec(name) is not None
                       for name in ['numpy', 'torch', 'onnxruntime', 'rclpy', 'cv2']},
}, indent=2), flush=True)
