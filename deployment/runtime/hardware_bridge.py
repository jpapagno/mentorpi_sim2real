"""Exclusive board owner: sensor publication and fail-closed motor watchdog.

Only /mentorpi/policy_cmd_vel plus a fresh motion_enable heartbeat can move it.
Vendor /cmd_vel and set_motor routes have no hardware subscription here.
"""
import fcntl
import json
import math
from pathlib import Path
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, UInt16, String
from drivetrain import DriveConfig
from rrc_serial import SerialBoard
from command_guard import CommandGuard


class HardwareBridge(Node):
    def __init__(self):
        super().__init__('mentorpi_hardware_bridge')
        self.guard=CommandGuard()
        self.lock=threading.Lock()
        self.running=True
        self.started=time.monotonic()
        self.last_reason=None
        self.last_enable={'value':False,'at':time.monotonic()}
        self.enable_counts={'true':0,'false':0}
        config=Path('/home/ubuntu/shared/mentorpi_nav/config/drivetrain.json')
        self.kinematics=DriveConfig(**json.loads(config.read_text()))
        self.wheel_speeds=(0.,)*4
        self.body_command=(0.,)*3
        self.board=SerialBoard(timeout=.01)
        self.board.enable_reception()
        self.board.set_motor_speed([[i,0.] for i in range(1,5)])
        self.imu_pub=self.create_publisher(Imu,'/ros_robot_controller/imu_raw',10)
        self.battery_pub=self.create_publisher(UInt16,'/ros_robot_controller/battery',10)
        self.applied_pub=self.create_publisher(Twist,'/mentorpi/applied_cmd_vel',10)
        # Legacy diagnostic alias only; no command-integrated odometry is fused.
        self.odom_command_pub=self.create_publisher(Twist,'/controller/cmd_vel',10)
        self.status_pub=self.create_publisher(String,'/mentorpi/guard_status',10)
        self.create_subscription(Twist,'/mentorpi/policy_cmd_vel',self.command,10)
        self.create_subscription(Bool,'/mentorpi/motion_enable',self.enable,10)
        self.create_subscription(LaserScan,'/scan_raw',self.scan,qos_profile_sensor_data)
        self.create_timer(.02,self.sensors)
        self.watchdog=threading.Thread(target=self.motor_loop,daemon=True)
        self.watchdog.start()
        self.get_logger().info('Hardware bridge started DISARMED; legacy motor topics are not connected')

    def command(self,msg):
        with self.lock:self.guard.receive_command((msg.linear.x,msg.linear.y,msg.angular.z),time.monotonic())

    def enable(self,msg):
        with self.lock:
            self.guard.receive_enable(msg.data,time.monotonic())
            self.last_enable={'value':bool(msg.data),'at':time.monotonic()}
            self.enable_counts['true' if msg.data else 'false']+=1

    def scan(self,msg):
        ranges=np.asarray(msg.ranges)
        valid=ranges[np.isfinite(ranges)&(ranges>=msg.range_min)&(ranges<=msg.range_max)]
        age=self.get_clock().now().nanoseconds*1e-9-(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
        if len(valid)<20 or not -.05<=age<.4:return
        # Track scan freshness; clearance is diagnostic, not a motor override.
        with self.lock:
            self.guard.scan_at=time.monotonic()
            self.guard.clearance=float(valid.min())

    def sensors(self):
        data=self.board.get_imu()
        if data is not None:
            ax,ay,az,gx,gy,gz=data
            msg=Imu()
            msg.header.frame_id='imu_link'
            msg.header.stamp=self.get_clock().now().to_msg()
            msg.orientation_covariance[0]=-1.
            msg.linear_acceleration.x=ax*9.80665
            msg.linear_acceleration.y=ay*9.80665
            msg.linear_acceleration.z=az*9.80665
            msg.angular_velocity.x=math.radians(gx)
            msg.angular_velocity.y=math.radians(gy)
            msg.angular_velocity.z=math.radians(gz)
            msg.angular_velocity_covariance=[.01,0.,0.,0.,.01,0.,0.,0.,.01]
            msg.linear_acceleration_covariance=[.01,0.,0.,0.,.01,0.,0.,0.,.01]
            self.imu_pub.publish(msg)
            with self.lock:self.guard.imu_at=time.monotonic()
        elif time.monotonic()-self.started>10 and time.monotonic()-self.guard.imu_at>2:
            self.get_logger().error('Control-board sensor stream lost; restarting the disarmed stack')
            self.running=False;rclpy.try_shutdown();return
        battery=self.board.get_battery()
        if battery is not None:
            msg=UInt16();msg.data=battery;self.battery_pub.publish(msg)

    def motor_loop(self):
        failures=0
        previous=time.monotonic()
        while self.running:
            with self.lock:command,reasons=self.guard.output(time.monotonic())
            try:
                now=time.monotonic();dt=max(.001,min(.05,now-previous));previous=now
                self.body_command=self.kinematics.advance_body(self.body_command,command,dt,stop=bool(reasons))
                target=self.kinematics.wheel_speeds(self.body_command)
                self.wheel_speeds=self.kinematics.advance(self.wheel_speeds,target,dt,stop=bool(reasons))
                self.board.set_motor_speed([[index+1,rps] for index,rps in enumerate(self.wheel_speeds)])
                applied=self.kinematics.body_velocity(self.wheel_speeds)
                failures=0
                msg=Twist();msg.linear.x=applied[0];msg.linear.y=applied[1];msg.angular.z=applied[2]
                self.applied_pub.publish(msg)
                self.odom_command_pub.publish(msg)
                status=String();status.data=json.dumps({'applied':applied,'wheel_command_rps':self.wheel_speeds,
                    'velocity_is_measured':False,'serial_rx_bytes':self.board.rx_bytes,
                    'serial_valid_packets':self.board.parser.valid,'blocked_by':reasons})
                current=json.loads(status.data)
                current.update(last_enable=self.last_enable,enable_counts=self.enable_counts,serial_tx_bytes=self.board.tx_bytes)
                status.data=json.dumps(current)
                self.status_pub.publish(status)
            except Exception as exc:
                with self.lock:self.guard.enabled=False
                self.wheel_speeds=(0.,)*4
                self.body_command=(0.,)*3
                self.get_logger().error(f'Hardware output failed: {exc}')
                failures+=1
                if failures>=3:
                    self.running=False;rclpy.try_shutdown();break
            time.sleep(.02)

    def stop(self):
        with self.lock:self.guard.enabled=False
        self.running=False
        self.watchdog.join(timeout=1.)
        try:self.board.set_motor_speed([[i,0.] for i in range(1,5)])
        finally:
            self.board.enable_reception(False)
            self.board.port.close()


def main():
    root=Path('/home/ubuntu/shared/mentorpi_nav')
    lock=(root/'hardware.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rclpy.init()
    node=HardwareBridge()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
