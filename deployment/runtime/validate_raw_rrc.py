"""Compare independent packets with installed SDK, using a fake serial port."""
import json
import sys
import struct
sys.path.insert(0, '/home/ubuntu/shared/mentorpi_nav')
from raw_rrc_probe import motors, packet
from ros_robot_controller.ros_robot_controller_sdk import Board

class FakePort:
    def __init__(self): self.writes = []
    def write(self, value): self.writes.append(bytes(value)); return len(value)

board = Board.__new__(Board)
board.port = FakePort()
cases = 0
for values in ([0.,0.,0.,0.], [.6,0.,0.,0.], [0.,-.6,0.,0.], [0.,0.,.6,0.], [0.,0.,0.,-.6]):
    board.set_motor_speed([[i+1,v] for i,v in enumerate(values)])
    assert board.port.writes[-1] == motors(values)
    cases += 1
board.set_buzzer(2000,.1,.1,1)
assert board.port.writes[-1] == packet(2,struct.pack('<HHHH',2000,100,100,1))
print(json.dumps({'passed':True,'motor_cases':cases,'buzzer_case':True,'serial_ports_opened':0}),flush=True)
