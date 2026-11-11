"""Recover only this launch's stale LiDAR driver; navigation stays disarmed on faults."""
import os
from pathlib import Path
import signal
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from .paths import CONFIG


class SensorSupervisor(Node):
    def __init__(self):
        super().__init__('mentorpi_sensor_supervisor')
        self.last=time.monotonic();self.last_restart=time.monotonic()
        self.create_subscription(LaserScan,'/scan_raw',self.scan,qos_profile_sensor_data)
        self.create_timer(1.,self.check)

    def scan(self,msg):self.last=time.monotonic()

    def check(self):
        if time.monotonic()-self.last<2 or time.monotonic()-self.last_restart<10:return
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args=path.read_bytes().split(b'\0')
                if args[0].endswith(b'/ldlidar_stl_ros2_node') and str(CONFIG/"lidar.yaml").encode() in args:
                    os.kill(int(path.parent.name),signal.SIGTERM)
                    self.get_logger().warning('Restarted owned stale LiDAR driver; hardware watchdog holds zero')
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        self.last_restart=time.monotonic()


def main():
    rclpy.init();node=SensorSupervisor()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__ == "__main__":main()
