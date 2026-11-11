"""Derive holonomic twist from measured laser poses, never motor commands."""
import copy
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from navigation_math import yaw_of
from motion_feedback import PlanarVelocityEstimator

class MeasuredOdometry(Node):
    def __init__(self):
        super().__init__('mentorpi_measured_odometry')
        self.estimator=PlanarVelocityEstimator()
        self.publisher=self.create_publisher(Odometry,'/mentorpi/measured_odom',10)
        self.create_subscription(Odometry,'/odom_rf2o',self.receive,10)
    def receive(self,msg):
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        velocity=self.estimator.add(stamp,p.x,p.y,yaw_of([q.x,q.y,q.z,q.w]))
        if velocity is None:return
        result=copy.deepcopy(msg)
        result.twist.twist.linear.x,result.twist.twist.linear.y,result.twist.twist.angular.z=velocity
        result.twist.covariance[0]=.01;result.twist.covariance[7]=.01;result.twist.covariance[35]=.04
        result.pose.covariance[0]=.0025;result.pose.covariance[7]=.0025;result.pose.covariance[35]=.01
        self.publisher.publish(result)
def main():
    rclpy.init();node=MeasuredOdometry()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
