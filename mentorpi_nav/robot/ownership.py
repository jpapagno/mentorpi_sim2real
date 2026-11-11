"""Idle-disarmed ownership handoff for local policy runners; no saved-map dependency."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

class PolicyOwnership(Node):
    def __init__(self):
        super().__init__('mentorpi_policy_ownership')
        self.paused=False
        self.publisher=self.create_publisher(Bool,'/mentorpi/motion_enable',10)
        self.create_service(SetBool,'/mentorpi/policy_ownership',self.ownership)
        self.create_timer(.05,self.idle)
    def idle(self):
        if not self.paused:
            message=Bool();message.data=False;self.publisher.publish(message)
    def ownership(self,request,response):
        message=Bool();message.data=False;self.publisher.publish(message)
        self.paused=bool(request.data)
        response.success=True
        response.message='Local policy controller ownership' if self.paused else 'Returned to disarmed local supervisor'
        return response
def main():
    rclpy.init();node=PolicyOwnership()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
