"""Continuous filtered depth obstacles and inspectable binary policy map."""
import json
import time
import numpy as np
import rclpy
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header,String
from navigation_io import NavigationIO


class PerceptionNode(NavigationIO):
    def __init__(self):
        super().__init__('mentorpi_perception')
        self.cloud_pub=self.create_publisher(PointCloud2,'/mentorpi/depth_obstacles',10)
        self.grid_pub=self.create_publisher(OccupancyGrid,'/mentorpi/policy_grid',1)
        self.health_pub=self.create_publisher(String,'/mentorpi/perception_health',10)
        self.last_depth_stamp=None
        self.create_timer(.1,self.update)
        self.last_grid=0.

    def update(self):
        try:
            if self.depth is not None and self.intrinsics is not None:
                stamp=(self.depth.header.stamp.sec,self.depth.header.stamp.nanosec)
                if stamp!=self.last_depth_stamp and self.age(self.depth)<.5:
                    points=self.depth_points().astype(np.float32)
                    header=Header();header.frame_id='base_footprint';header.stamp=self.depth.header.stamp
                    self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header,points))
                    self.last_depth_stamp=stamp
            health=self.health()
            status=String();status.data=json.dumps({'blocked_by':health})
            self.health_pub.publish(status)
            if not health and time.monotonic()-self.last_grid>.2:
                base,_=self.pose()
                _,grid,_=self.observations(base[:2],[0.,0.,0.])
                msg=OccupancyGrid();msg.header.frame_id='base_footprint';msg.header.stamp=self.get_clock().now().to_msg()
                msg.info.resolution=.05;msg.info.width=100;msg.info.height=100
                msg.info.origin.position.x=-2.5;msg.info.origin.position.y=-2.5;msg.info.origin.orientation.w=1.
                msg.data=(grid[0,0,::-1,::-1].T.flatten()*100).astype(np.int8).tolist()
                self.grid_pub.publish(msg)
                self.last_grid=time.monotonic()
        except Exception as exc:
            self.get_logger().warning(str(exc),throttle_duration_sec=5.)


def main():
    rclpy.init();node=PerceptionNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
