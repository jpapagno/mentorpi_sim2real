"""Read-only ROS topic inventory with measured arrivals and compact samples."""
from array import array
import json
from pathlib import Path
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

def compact(value, depth=0):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:500] if isinstance(value, str) else value
    if depth > 7:
        return str(type(value).__name__)
    if hasattr(value, 'get_fields_and_field_types'):
        return {key: compact(getattr(value, key), depth+1) for key in value.get_fields_and_field_types()}
    if isinstance(value, (list, tuple, array)) or hasattr(value, 'tolist'):
        return {'length': len(value), 'first_values': [compact(v, depth+1) for v in value[:10]]}
    return str(value)[:300]

rclpy.init()
node = rclpy.create_node('mentorpi_readonly_topic_snapshot')
for _ in range(30):
    rclpy.spin_once(node, timeout_sec=.1)
report = {'drive_commands_published': 0, 'topics': {}}
subscriptions = []
arrivals = {}
last = {}
for topic, types in node.get_topic_names_and_types():
    if topic.startswith('/mentorpi_readonly'):
        continue
    pubs = node.get_publishers_info_by_topic(topic)
    subs = node.get_subscriptions_info_by_topic(topic)
    report['topics'][topic] = {'types': types,
        'publishers': [{'node': p.node_namespace.rstrip('/')+'/'+p.node_name,
                        'reliability': str(p.qos_profile.reliability),
                        'durability': str(p.qos_profile.durability)} for p in pubs],
        'subscribers': [p.node_namespace.rstrip('/')+'/'+p.node_name for p in subs]}
    arrivals[topic] = []
    def receive(msg, name=topic):
        arrivals[name].append(time.monotonic())
        last[name] = msg
    try:
        subscriptions.append(node.create_subscription(get_message(types[0]), topic, receive, qos_profile_sensor_data))
    except Exception as exc:
        report['topics'][topic]['subscription_error'] = str(exc)
start = time.monotonic()
while time.monotonic()-start < 12:
    rclpy.spin_once(node, timeout_sec=.02)
report['window_s'] = time.monotonic()-start
for topic, info in report['topics'].items():
    stamps = arrivals[topic]
    info['messages'] = len(stamps)
    info['observed_hz'] = (len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else 0
    if topic in last:
        msg = last[topic]
        if hasattr(msg, 'header'):
            info['last_stamp_age_s'] = node.get_clock().now().nanoseconds/1e9-msg.header.stamp.sec-msg.header.stamp.nanosec/1e9
        if topic.endswith('/image') or topic.endswith('/image_raw') or topic.endswith('/points'):
            info['sample'] = {key: compact(getattr(msg,key)) for key in msg.get_fields_and_field_types() if key != 'data'}
            info['sample']['data_bytes'] = len(msg.data)
        else:
            info['sample'] = compact(msg)
out = Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/topic_snapshot.json')
out.write_text(json.dumps(report, indent=2))
print(json.dumps({t: {'messages': i['messages'], 'hz': round(i['observed_hz'],2), 'publishers':len(i['publishers']), 'subscribers':len(i['subscribers'])} for t,i in report['topics'].items()}, indent=2), flush=True)
node.destroy_node()
rclpy.shutdown()
