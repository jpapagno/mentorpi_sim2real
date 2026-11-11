"""Stop-only teardown helper. It has no interface for nonzero commands."""
import time
from rrc_serial import SerialBoard

board=SerialBoard(timeout=.01)
try:
    for _ in range(3):
        board.set_motor_speed([[index,0.] for index in range(1,5)])
        time.sleep(.02)
finally:board.port.close()
