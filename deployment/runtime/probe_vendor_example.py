"""Run stock Hiwonder controller with one bounded, supervised velocity pulse."""
import json
import os
import signal
import subprocess
import time
import rclpy
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import MotorsState, MotorState

process = None
node = None
samples = []
log = open('/home/ubuntu/shared/mentorpi_nav/logs/vendor_example.log', 'w')
try:
    process = subprocess.Popen(['ros2', 'launch', 'controller', 'controller.launch.py',
                                'enable_odom:=false'], stdout=log, stderr=log,
                               start_new_session=True)
    rclpy.init()
    node = rclpy.create_node('supervised_vendor_example_probe')
    pub = node.create_publisher(Twist, '/controller/cmd_vel', 10)
    stop_pub = node.create_publisher(MotorsState, '/ros_robot_controller/set_motor', 10)
    node.create_subscription(MotorsState, '/ros_robot_controller/set_motor',
                             lambda msg: samples.append([[m.id, m.rps] for m in msg.data]), 10)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
        if pub.get_subscription_count() and stop_pub.get_subscription_count() >= 2:
            break
        if process.poll() is not None:
            raise RuntimeError('Vendor launch exited')
    else:
        raise RuntimeError('Vendor command/motor path not ready')
    print('VENDOR_READY: forward 0.1 m/s for 0.5 seconds', flush=True)
    msg = Twist()
    msg.linear.x = .1
    deadline = time.monotonic() + .5
    while time.monotonic() < deadline:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=.02)
finally:
    if node is not None:
        zeros = MotorsState()
        zeros.data = [MotorState(id=i, rps=0.) for i in range(1, 5)]
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            pub.publish(Twist())
            stop_pub.publish(zeros)
            rclpy.spin_once(node, timeout_sec=.02)
        print(json.dumps({'nonzero_motor_messages': sum(any(abs(m[1]) > 1e-6 for m in row) for row in samples),
                          'last_motor_message': samples[-1] if samples else None,
                          'zero_stop_published': True}), flush=True)
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
    if node is not None:
        node.destroy_node()
        rclpy.shutdown()
    log.close()
