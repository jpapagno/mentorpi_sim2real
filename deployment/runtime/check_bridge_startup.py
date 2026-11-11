"""Exercise new motor bridge startup/teardown without publishing drive commands."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import rclpy
from sensor_msgs.msg import Imu
from std_msgs.msg import String,UInt16

root=Path('/home/ubuntu/shared/mentorpi_nav')
log=(root/'logs/independent_bridge_check.log').open('w')
child=subprocess.Popen([sys.executable,str(root/'runtime/hardware_bridge.py')],stdout=log,stderr=log)
rclpy.init()
node=rclpy.create_node('mentorpi_disarmed_bridge_check')
statuses=[]
imus=[]
batteries=[]
node.create_subscription(String,'/mentorpi/guard_status',lambda m:statuses.append(json.loads(m.data)),10)
node.create_subscription(Imu,'/ros_robot_controller/imu_raw',lambda m:imus.append(m.header.stamp.sec),10)
node.create_subscription(UInt16,'/ros_robot_controller/battery',lambda m:batteries.append(m.data),10)
try:
    deadline=time.monotonic()+6
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.02)
        if child.poll() is not None:break
finally:
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill();child.wait()
    log.close()
    node.destroy_node();rclpy.shutdown()
    from rrc_serial import SerialBoard
    board=SerialBoard()
    try:board.set_motor_speed([[i,0.] for i in range(1,5)])
    finally:board.port.close()
result={'guard_samples':len(statuses),'imu_samples':len(imus),'battery_mv':batteries[-3:],
        'max_reported_command':max((max(abs(v) for v in s['applied']) for s in statuses),default=None),
        'final_guard':statuses[-1] if statuses else None,'drive_commands_published':0,
        'bridge_exit_code':child.returncode}
result['passed']=bool(len(statuses)>10 and len(imus)>10 and batteries and
    result['max_reported_command']==0 and 'disarmed' in statuses[-1]['blocked_by'] and child.returncode==0)
(root/'diagnostics/independent_bridge_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
raise SystemExit(0 if result['passed'] else 1)
