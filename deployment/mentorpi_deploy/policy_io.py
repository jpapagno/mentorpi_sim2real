"""Motor command publication and ownership for the policy runner."""

import fcntl
import json
import time
from contextlib import contextmanager

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .navigation_io import NavigationIO
from .paths import ROOT


class PolicyIO(NavigationIO):
    def __init__(self):
        super().__init__("mentorpi_policy")
        self.command_pub = self.create_publisher(Twist, "/mentorpi/policy_cmd_vel", 10)
        self.enable_pub = self.create_publisher(Bool, "/mentorpi/motion_enable", 10)
        self.status = {}
        self.ownership = self.create_client(SetBool, "/mentorpi/policy_ownership")
        self.create_subscription(String, "/mentorpi/guard_status", self.on_status, 10)

    def on_status(self, msg):
        self.status = json.loads(msg.data)

    def publish(self, velocity, enabled):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.linear.y = float(velocity[1])
        msg.angular.z = float(velocity[2])
        enable = Bool()
        enable.data = enabled
        self.enable_pub.publish(enable)
        self.command_pub.publish(msg)

    def phase(self, seconds, velocity, enabled):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.publish(velocity, enabled)
            until = min(deadline, time.monotonic() + 0.05)
            while time.monotonic() < until:
                rclpy.spin_once(self, timeout_sec=0.005)

    def set_ownership(self, enabled):
        request = SetBool.Request()
        request.data = enabled
        future = self.ownership.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3)
        if not future.done() or not future.result().success:
            raise RuntimeError("Policy ownership request failed")


@contextmanager
def control_session():
    """Serialize control and calibration, and disarm before releasing ownership."""
    with (ROOT / "policy_session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another policy or calibration session is active") from exc
        rclpy.init()
        node = None
        try:
            node = PolicyIO()
            if not node.ownership.wait_for_service(timeout_sec=20):
                raise RuntimeError("Ownership service unavailable")
            try:
                node.set_ownership(True)
                yield node
            finally:
                if rclpy.ok():
                    try:
                        node.phase(0.5, (0.0, 0.0, 0.0), False)
                    finally:
                        if node.ownership.service_is_ready():
                            node.set_ownership(False)
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.try_shutdown()
