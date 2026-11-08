"""Inspect actual installed wheel/packet encoding without opening serial ports."""
import inspect
import json
from controller.mecanum import MecanumChassis
from ros_robot_controller.ros_robot_controller_sdk import Board

wheel_message=MecanumChassis(wheelbase=.1368,track_width=.1446,wheel_diameter=.065).set_velocity(.1,0.,0.)
wheels=[[wheel.id,wheel.rps] for wheel in wheel_message.data]
fake=Board.__new__(Board)
captured=[]
fake.buf_write=lambda function,data:captured.append({'function':int(function),'data_hex':bytes(data).hex()})
fake.set_motor_speed(wheels)
print(json.dumps({'kinematics_module':inspect.getfile(MecanumChassis),
 'sdk_module':inspect.getfile(Board),'wheels_at_0_1_mps':wheels,'packet_payload':captured,
 'installed_set_motor_speed_source':inspect.getsource(Board.set_motor_speed),
 'installed_buf_write_source':inspect.getsource(Board.buf_write)},indent=2),flush=True)
