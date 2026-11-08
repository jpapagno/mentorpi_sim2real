import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from scan_math import fixed_circle_scan


class ScanNormalizer(Node):
    def __init__(self):
        super().__init__('mentorpi_scan_normalizer')
        self.publisher=self.create_publisher(LaserScan,'/scan',10)
        self.create_subscription(LaserScan,'/scan_raw',self.receive,qos_profile_sensor_data)

    def receive(self,msg):
        try:ranges=fixed_circle_scan(msg.ranges,msg.angle_min,msg.angle_increment,msg.range_min,msg.range_max)
        except ValueError as exc:
            self.get_logger().error(str(exc),throttle_duration_sec=5.)
            return
        out=LaserScan()
        out.header=msg.header
        out.angle_min=-math.pi
        out.angle_increment=2*math.pi/len(ranges)
        out.angle_max=out.angle_min+(len(ranges)-1)*out.angle_increment
        out.scan_time=msg.scan_time
        out.time_increment=msg.scan_time/len(ranges)
        out.range_min=msg.range_min
        out.range_max=msg.range_max
        out.ranges=ranges.tolist()
        self.publisher.publish(out)


def main():
    rclpy.init();node=ScanNormalizer()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
