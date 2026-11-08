"""Stop-only teardown helper. It has no interface for nonzero commands."""
import time
from ros_robot_controller.ros_robot_controller_sdk import Board

board=Board(timeout=.05)
try:
    for _ in range(3):
        board.set_motor_speed([[index,0.] for index in range(1,5)])
        time.sleep(.02)
finally:board.port.close()
