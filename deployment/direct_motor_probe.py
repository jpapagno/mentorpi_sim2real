"""Optional bounded laptop-to-board diagnostic, independent of the Pi/ROS.

Default only lists the exact previously identified board. A pulse requires that
board, explicit --pulse, and a current observer recording readiness file.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import serial
from serial.tools import list_ports

ROBOT_SERIAL='REPLACE_WITH_YOUR_BOARD_SERIAL'


def crc8(data):
    value=0
    for byte in data:
        value^=byte
        for _ in range(8):value=(value>>1)^0x8c if value&1 else value>>1
    return value


def motor_packet(speeds):
    payload=bytes([1,4])+b''.join(struct.pack('<Bf',index,float(speed)) for index,speed in enumerate(speeds))
    body=bytes([3,len(payload)])+payload
    return b'\xaa\x55'+body+bytes([crc8(body)])


def wheel_speeds(axis,speed):
    x,y,w=(speed,0.,0.) if axis=='x' else (0.,speed,0.) if axis=='y' else (0.,0.,speed)
    turn=w*(.1368+.1446)/2
    return tuple(value/(math.pi*.065) for value in (-(x-y-turn),-(x+y-turn),x+y+turn,x-y+turn))


def open_board(port):
    connection=serial.Serial(None,1000000,timeout=.1,write_timeout=.2)
    connection.rts=False;connection.dtr=False;connection.port=port;connection.open()
    return connection


def stop(port):
    with open_board(port) as connection:
        for _ in range(3):connection.write(motor_packet([0.,0.,0.,0.]));time.sleep(.03)


def worker(port,axis,speed,duration):
    with open_board(port) as connection:
        try:
            for _ in range(10):connection.write(motor_packet([0.,0.,0.,0.]));time.sleep(.05)
            deadline=time.monotonic()+duration
            while time.monotonic()<deadline:
                connection.write(motor_packet(wheel_speeds(axis,speed)));time.sleep(.02)
        finally:
            for _ in range(3):connection.write(motor_packet([0.,0.,0.,0.]));time.sleep(.03)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pulse',choices=['x','y','yaw'])
    parser.add_argument('--speed',type=float,default=.05)
    parser.add_argument('--duration',type=float,default=.5)
    parser.add_argument('--observer-ready')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--port',help=argparse.SUPPRESS)
    args=parser.parse_args()
    matches=[port for port in list_ports.comports() if port.serial_number==ROBOT_SERIAL and port.vid==0x1a86]
    if not args.worker:
        print(json.dumps({'matching_robot_ports':[{'port':p.device,'description':p.description,'serial':p.serial_number} for p in matches]}),flush=True)
    if not args.pulse:return 0
    if not math.isfinite(args.speed) or not math.isfinite(args.duration) or not 0<args.speed<=.1 or not 0<args.duration<=1. or args.speed*args.duration>.05:
        raise ValueError('Pulse exceeds bounded commissioning limits')
    if len(matches)!=1:raise RuntimeError('Exact robot control board is not uniquely connected; no commands sent')
    port=matches[0].device
    if args.worker:
        if args.port!=port:raise RuntimeError('Board identity changed')
        worker(port,args.pulse,args.speed,args.duration);return 0
    if not args.observer_ready:raise RuntimeError('A current observer recording is required')
    observer=Path(args.observer_ready)
    if not observer.exists() or time.time()-observer.stat().st_mtime>3:
        raise RuntimeError('Observer recording is not current')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--port',port,'--pulse',args.pulse,
             '--speed',str(args.speed),'--duration',str(args.duration)]
    process=subprocess.Popen(command,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        code=process.wait(timeout=3.)
        if code:raise RuntimeError(f'Pulse worker exited {code}')
    except subprocess.TimeoutExpired:
        process.kill();process.wait();raise RuntimeError('Worker deadline exceeded; stop-only recovery sent')
    finally:stop(port)
    print('Bounded pulse ended; motor-zero packets sent. Camera review still required.',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

