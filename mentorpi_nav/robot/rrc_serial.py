"""Small RRC Lite transport: CRC-framed serial, motor writes and sensor reads.

No ROS or vendor SDK. Successful writes are not motor acknowledgements.
"""
import math
import os
from pathlib import Path
import struct
import threading

DEVICE = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5917011712-if00'

def serial_owners(path):
    """Find visible open descriptors without requiring fuser in the container.

Host-side service preflight additionally checks processes outside the container.
"""
    target=os.stat(path).st_rdev
    owners=set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (process/'fd').iterdir():
                try:
                    if fd.stat().st_rdev == target:
                        owners.add(int(process.name))
                except (FileNotFoundError,PermissionError):
                    pass
        except (FileNotFoundError,PermissionError):
            pass
    return sorted(owners)

def crc8(data):
    value = 0
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x8c if value & 1 else 0)
    return value

def packet(function, payload):
    if not 0 <= function <= 11 or len(payload) > 255:
        raise ValueError('Invalid RRC packet')
    body = bytes([function,len(payload)])+bytes(payload)
    return b'\xaa\x55'+body+bytes([crc8(body)])

def motor_packet(speeds):
    if len(speeds) != 4 or not all(math.isfinite(v) for v in speeds):
        raise ValueError('Four finite motor speeds required')
    return packet(3,b'\x01\x04'+b''.join(struct.pack('<Bf',i,float(v)) for i,v in enumerate(speeds)))

class PacketParser:
    def __init__(self):
        self.buffer = bytearray()
        self.valid = 0
        self.bad_crc = 0

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 5:
            if self.buffer[:2] != b'\xaa\x55' or self.buffer[2] > 11:
                del self.buffer[0]
                continue
            size = 5+self.buffer[3]
            if len(self.buffer) < size:
                break
            if crc8(self.buffer[2:size-1]) != self.buffer[size-1]:
                self.bad_crc += 1
                del self.buffer[0]
                continue
            frames.append((self.buffer[2],bytes(self.buffer[4:size-1])))
            self.valid += 1
            del self.buffer[:size]
        return frames

class SerialBoard:
    def __init__(self, timeout=.01, device=DEVICE):
        import serial
        path = Path(device)
        if not path.exists():
            raise RuntimeError('Recorded RRC Lite board is absent')
        owners=serial_owners(path)
        if owners:
            raise RuntimeError('RRC serial port already owned: '+str(owners))
        self.port = serial.Serial(None,1000000,timeout=timeout,write_timeout=.2,exclusive=True)
        self.port.rts = False
        self.port.dtr = False
        self.port.port = str(path)
        self.port.open()
        # Enforce kernel exclusivity even against clients that ignore flock.
        import fcntl
        import termios
        try:
            fcntl.ioctl(self.port.fileno(),termios.TIOCEXCL)
        except Exception:
            self.port.close()
            raise
        self.parser = PacketParser()
        self.lock = threading.Lock()
        self.enabled = False
        self.imu = None
        self.battery = None
        self.rx_bytes = 0
        self.tx_bytes = 0

    def send(self, data):
        with self.lock:
            count = self.port.write(data)
            if count != len(data):
                raise IOError('Incomplete serial write')
            self.tx_bytes += count

    def set_motor_speed(self, channels):
        if sorted(row[0] for row in channels) != [1,2,3,4]:
            raise ValueError('All four motor channels required')
        ordered = sorted(channels,key=lambda row:row[0])
        self.send(motor_packet([row[1] for row in ordered]))

    def enable_reception(self, enabled=True):
        self.enabled = enabled

    def poll(self):
        if not self.enabled:
            return
        waiting = self.port.in_waiting
        if not waiting:
            return
        data = self.port.read(min(waiting,65536))
        self.rx_bytes += len(data)
        for function,payload in self.parser.feed(data):
            if function == 7 and len(payload) == 24:
                values = struct.unpack('<6f',payload)
                if all(math.isfinite(v) for v in values):
                    self.imu = values
            elif function == 0 and len(payload) == 3 and payload[0] == 4:
                self.battery = struct.unpack('<H',payload[1:])[0]

    def get_imu(self):
        self.poll()
        value,self.imu = self.imu,None
        return value

    def get_battery(self):
        self.poll()
        value,self.battery = self.battery,None
        return value
