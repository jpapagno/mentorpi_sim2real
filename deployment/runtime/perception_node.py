"""Publish the inspectable LiDAR-only binary policy grid."""
import json
import time
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from navigation_io import NavigationIO


class PerceptionNode(NavigationIO):
    def __init__(self):
        super().__init__('mentorpi_perception')
        self.grid_pub=self.create_publisher(OccupancyGrid,'/mentorpi/policy_grid',1)
        self.health_pub=self.create_publisher(String,'/mentorpi/perception_health',10)
        self.create_timer(.1,self.update)
        self.last_grid=0.

    def update(self):
        try:
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
