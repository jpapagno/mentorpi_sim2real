"""Independent RRC serial diagnostic. No ROS or Hiwonder SDK imports.

Default sends only zero motor commands and records incoming packet types.
A motor pulse requires an explicit channel and raised-wheel acknowledgement.
"""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import serial

DEVICE = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5917011712-if00'
ROOT = Path('/home/pi/docker/tmp/mentorpi_nav')

def crc8(data):
    value = 0
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x8c if value & 1 else 0)
    return value

def packet(function, payload):
    body = bytes([function, len(payload)]) + payload
    return b'\xaa\x55' + body + bytes([crc8(body)])

def motors(speeds):
    return packet(3, bytes([1, 4]) + b''.join(struct.pack('<Bf', i, float(v)) for i,v in enumerate(speeds)))

def connect():
    port = serial.Serial(None, 1000000, timeout=.05, write_timeout=.2)
    port.rts = False
    port.dtr = False
    port.port = DEVICE
    port.open()
    return port

def write(port, data, records):
    count = port.write(data)
    port.flush()
    records.append({'hex':data.hex(), 'requested_bytes':len(data), 'written_bytes':count})
    if count != len(data):
        raise RuntimeError('Incomplete serial write')

def stop():
    with connect() as port:
        for _ in range(5):
            port.write(motors([0]*4))
            port.flush()
            time.sleep(.02)

def worker(motor, rps, duration, beep):
    records = []
    raw = bytearray()
    with connect() as port:
        try:
            write(port, motors([0]*4), records)
            if beep:
                write(port, packet(2, struct.pack('<HHHH', 2000, 100, 100, 1)), records)
            deadline = time.monotonic()+1
            while time.monotonic()<deadline:
                raw.extend(port.read(4096))
            if motor:
                speeds = [0.]*4
                speeds[motor-1] = rps
                deadline = time.monotonic()+duration
                while time.monotonic()<deadline:
                    write(port, motors(speeds), records)
                    time.sleep(.02)
        finally:
            for _ in range(5):
                write(port, motors([0]*4), records)
                time.sleep(.02)
    packets = Counter()
    offset = 0
    while offset+5 <= len(raw):
        if raw[offset:offset+2] != b'\xaa\x55':
            offset += 1
            continue
        end = offset+5+raw[offset+3]
        if end > len(raw):
            break
        body = raw[offset+2:end-1]
        if crc8(body) == raw[end-1]:
            packets[str(raw[offset+2])] += 1
            offset = end
        else:
            offset += 1
    result = {'device':DEVICE, 'baud':1000000, 'motor':motor, 'rps':rps if motor else 0,
              'duration_s':duration if motor else 0, 'rx_bytes':len(raw),
              'valid_received_packet_functions':dict(packets), 'writes':records,
              'physical_response':'requires human observation; serial write is not motor acknowledgement'}
    (ROOT/'diagnostics/raw_rrc_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motor',type=int,choices=range(1,5))
    parser.add_argument('--rps',type=float,default=.6)
    parser.add_argument('--duration',type=float,default=.5)
    parser.add_argument('--wheels-raised',action='store_true')
    parser.add_argument('--beep',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 < args.rps <= .8 and 0 < args.duration <= .75):
        raise ValueError('Pulse outside diagnostic bounds')
    if args.motor and not args.wheels_raised:
        raise RuntimeError('Single-motor test requires raised wheels')
    if args.worker:
        worker(args.motor,args.rps,args.duration,args.beep)
        return
    lock = (ROOT/'hardware.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for service in ('start_node.service','mentorpi-navigation.service'):
        if subprocess.run(['systemctl','is-active','--quiet',service]).returncode == 0:
            raise RuntimeError('Stop conflicting service: '+service)
    if not Path(DEVICE).exists():
        raise RuntimeError('Recorded control board not found')
    # Refuse concurrent serial ownership before opening the port.
    if subprocess.run(['fuser',str(Path(DEVICE).resolve())],capture_output=True).returncode == 0:
        raise RuntimeError('Serial board already open by another process')
    command = [sys.executable,__file__,'--worker','--rps',str(args.rps),'--duration',str(args.duration)]
    if args.motor:
        command += ['--motor',str(args.motor),'--wheels-raised']
    if args.beep:
        command.append('--beep')
    child = subprocess.Popen(command)
    try:
        code = child.wait(timeout=5)
        if code:
            raise RuntimeError('Serial worker failed: '+str(code))
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        raise RuntimeError('Serial worker exceeded deadline')
    finally:
        stop()
        print('PARENT_STOP: explicit zero packets sent',flush=True)

if __name__ == '__main__':
    main()
