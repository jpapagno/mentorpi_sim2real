"""Ask AMCL to relocalize using stationary scans; publishes no drive commands."""
import json
import time
import rclpy
from std_srvs.srv import Empty
from navigation_io import NavigationIO

rclpy.init()
node = NavigationIO('mentorpi_stationary_relocalization')
try:
    for name, count in [('/reinitialize_global_localization', 1), ('/request_nomotion_update', 12)]:
        client = node.create_client(Empty, name)
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('Unavailable: ' + name)
        for _ in range(count):
            future = client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            if future.result() is None:
                raise RuntimeError('No response: ' + name)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.05)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    print(json.dumps({'health': node.health(), 'map_alignment': node.map_match,
                      'drive_commands_published': 0}, indent=2), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
