"""Read board sensor packets directly; never calls any motor/servo setter."""
import json
import time
from ros_robot_controller.ros_robot_controller_sdk import Board

board=Board(timeout=.2)
board.enable_reception()
samples=[]
battery=[]
started=time.monotonic()
while time.monotonic()-started<5:
    imu=board.get_imu()
    volts=board.get_battery()
    if imu is not None:samples.append(imu)
    if volts is not None:battery.append(volts)
    time.sleep(.005)
board.enable_reception(False)
board.port.close()
print(json.dumps({'imu_samples':len(samples),'last_imu':samples[-1] if samples else None,
                  'battery_packets':battery[-5:]}),flush=True)
raise SystemExit(0 if len(samples)>=3 else 1)
