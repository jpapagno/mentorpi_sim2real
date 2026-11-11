"""Motor command publication and ownership for the policy runner."""
import json
import time
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool,String
from std_srvs.srv import SetBool
from .navigation_io import NavigationIO


class PolicyIO(NavigationIO):
    def __init__(self):
        super().__init__('mentorpi_policy')
        self.command_pub=self.create_publisher(Twist,'/mentorpi/policy_cmd_vel',10)
        self.enable_pub=self.create_publisher(Bool,'/mentorpi/motion_enable',10)
        self.status={}
        self.ownership=self.create_client(SetBool,'/mentorpi/policy_ownership')
        self.create_subscription(String,'/mentorpi/guard_status',self.on_status,10)

    def on_status(self,msg):self.status=json.loads(msg.data)

    def publish(self,velocity,enabled):
        msg=Twist();msg.linear.x=float(velocity[0]);msg.linear.y=float(velocity[1]);msg.angular.z=float(velocity[2])
        enable=Bool();enable.data=enabled
        self.enable_pub.publish(enable);self.command_pub.publish(msg)

    def phase(self,seconds,velocity,enabled):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            self.publish(velocity,enabled)
            until=min(deadline,time.monotonic()+.05)
            while time.monotonic()<until:rclpy.spin_once(self,timeout_sec=.005)
