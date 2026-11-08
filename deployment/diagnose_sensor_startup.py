"""Targeted evidence for port/startup/power hypotheses; no motor commands."""
import glob
import json
from pathlib import Path
import subprocess


def capture(command):
    try:
        r=subprocess.run(command,shell=True,text=True,capture_output=True,timeout=15)
        return {'exit':r.returncode,'out':r.stdout,'err':r.stderr}
    except Exception as exc:return {'error':str(exc)}

paths=[
 '/home/ubuntu/ros2_ws/src/peripherals/launch/lidar.launch.py',
 '/home/ubuntu/ros2_ws/src/driver/ros_robot_controller/launch/ros_robot_controller.launch.py',
 '/home/ubuntu/ros2_ws/src/driver/controller/controller/odom_publisher_node.py',
 '/home/ubuntu/ros2_ws/src/driver/ros_robot_controller/ros_robot_controller/ros_robot_controller_node.py',
 '/home/ubuntu/ros2_ws/src/driver/ros_robot_controller/ros_robot_controller/ros_robot_controller_sdk.py',
]
logs=sorted(glob.glob('/home/ubuntu/.ros/log/*'),key=lambda p:Path(p).stat().st_mtime,reverse=True)
evidence={}
for directory in logs[:5]:
    for filename in glob.glob(directory+'/*.log'):
        lines=Path(filename).read_text(errors='replace').splitlines()
        signal=[line for line in lines if any(term in line.lower() for term in ['tty','serial','died','error','exception','traceback','ld19','controller_node'])]
        evidence[filename]=signal[-40:]
result={'commands':{
    'ports':capture('ls -l /dev/ttyUSB* /dev/ttyACM* /dev/rrc /dev/ldlidar /dev/lidar 2>/dev/null'),
    'bridge':capture('ros2 node info /ros_robot_controller; ros2 topic info /ros_robot_controller/imu_raw -v'),
    'driver_params':capture('ros2 param dump /LD19'),
    'maps':capture('ls -lh /home/ubuntu/ros2_ws/src/slam/maps; ls -lh /home/ubuntu/ros2_ws/src/navigation/config'),
    'versions':capture("python3 -c 'import torch,numpy; print(torch.__version__,numpy.__version__)'"),
},'files':{p:Path(p).read_text(errors='replace') for p in paths if Path(p).exists()},'logs':evidence}
print(json.dumps(result,indent=2),flush=True)
