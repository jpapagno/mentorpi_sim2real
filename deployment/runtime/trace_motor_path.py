"""Read-only timed trace of phone/ROS commands and motor board telemetry."""
import argparse
import json
from pathlib import Path
import time
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16
from ros_robot_controller_msgs.msg import MotorsState
from rclpy.qos import qos_profile_sensor_data

parser = argparse.ArgumentParser()
parser.add_argument('--duration', type=float, default=60)
args = parser.parse_args()
rclpy.init()
node = rclpy.create_node('mentorpi_readonly_motor_trace')
start = time.monotonic()
rows = []
subscriptions = []
def receive(topic, msg):
    if isinstance(msg, Twist):
        value = [msg.linear.x, msg.linear.y, msg.angular.z]
    elif isinstance(msg, MotorsState):
        value = [[m.id, m.rps] for m in msg.data]
    else:
        value = msg.data
    rows.append({'elapsed_s': round(time.monotonic()-start, 4), 'topic': topic, 'value': value})
for topic, kind in [('/cmd_vel', Twist), ('/controller/cmd_vel', Twist),
                    ('/ros_robot_controller/set_motor', MotorsState),
                    ('/ros_robot_controller/battery', UInt16)]:
    subscriptions.append(node.create_subscription(kind, topic, lambda msg,t=topic: receive(t,msg), qos_profile_sensor_data))
print('TRACE_READY: observing only; no drive commands published', flush=True)
try:
    while time.monotonic()-start < args.duration:
        rclpy.spin_once(node, timeout_sec=.05)
finally:
    result = {'drive_commands_published':0, 'duration_s':time.monotonic()-start, 'rows':rows}
    path = Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/motor_path_trace.json')
    path.write_text(json.dumps(result, indent=2))
    for topic in sorted({r['topic'] for r in rows}):
        values = [r for r in rows if r['topic']==topic]
        print(json.dumps({'topic':topic,'messages':len(values),'first':values[:3],'last':values[-3:]}),flush=True)
    node.destroy_node()
    rclpy.shutdown()
