"""Verify independent serial telemetry with zero-only motor output."""
import json
from pathlib import Path
import time
from rrc_serial import SerialBoard

board=SerialBoard()
counts={'imu':0,'battery':0}
battery=[]
last_imu=None
try:
    board.set_motor_speed([[i,0.] for i in range(1,5)])
    board.enable_reception()
    started=time.monotonic()
    while time.monotonic()-started<5:
        imu=board.get_imu()
        volts=board.get_battery()
        if imu is not None:
            counts['imu']+=1
            last_imu=imu
        if volts is not None:
            counts['battery']+=1
            battery.append(volts)
        time.sleep(.005)
finally:
    try:
        for _ in range(3):
            board.set_motor_speed([[i,0.] for i in range(1,5)])
            time.sleep(.02)
    finally:
        board.port.close()
result={'counts':counts,'last_imu':last_imu,'battery_mv':battery[-5:],
        'rx_bytes':board.rx_bytes,'tx_bytes':board.tx_bytes,
        'valid_packets':board.parser.valid,'bad_crc':board.parser.bad_crc,
        'nonzero_motor_commands':0,'passed':all(counts.values())}
Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/independent_transport.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
raise SystemExit(0 if result['passed'] else 1)
