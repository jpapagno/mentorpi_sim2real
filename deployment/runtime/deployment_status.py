"""Read-only final deployment/arm-state check; no command publishers."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from std_msgs.msg import String
from navigation_io import NavigationIO

parser=argparse.ArgumentParser()
parser.add_argument('--local',action='store_true',default=None)
args=parser.parse_args()
rclpy.init();node=NavigationIO('mentorpi_deployment_status_probe',local_mode=args.local)
guard=[]
node.create_subscription(String,'/mentorpi/guard_status',lambda msg:guard.append(json.loads(msg.data)),10)
deadline=time.monotonic()+20
while time.monotonic()<deadline:
    rclpy.spin_once(node,timeout_sec=.02)
    if len(guard)>=30 and not node.health():break
root=Path('/home/ubuntu/shared/mentorpi_nav')
legacy=node.get_subscriptions_info_by_topic('/ros_robot_controller/set_motor')
command_publishers=node.get_publishers_info_by_topic('/controller/cmd_vel')
acceptance=root/'diagnostics/commissioning_acceptance.json'
result={
    'navigation_mode':'local' if node.local_mode else 'global','pose_frame':node.pose_frame,
    'health':node.health(),'map_alignment':node.map_match,
    'observed_guard_messages':len(guard),
    'observed_applied_max':max((max(abs(v) for v in row['applied']) for row in guard),default=None),
    'guard_last':guard[-1] if guard else None,
    'legacy_motor_hardware_subscribers':len(legacy),
    'odometry_command_publishers':[info.node_name for info in command_publishers],
    'commissioning_acceptance_present':acceptance.exists(),
    'physical_trials_summary_present':(root/'logs/latest_trials.json').exists(),
    'model_selection':json.loads((root/'models/selection.json').read_text()),
    'model_sha256':hashlib.sha256((root/'models/policy.ts').read_bytes()).hexdigest(),
    'drive_commands_published_by_probe':0,
}
result['software_ready_disarmed']=bool(not result['health'] and guard and result['observed_applied_max']==0.
    and 'disarmed' in guard[-1].get('blocked_by',[]) and len(legacy)==0)
if not result['health']:result['pose']=node.pose()[0].tolist()
(root/'diagnostics/deployment_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
raise SystemExit(0 if result['software_ready_disarmed'] else 1)
