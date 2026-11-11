"""Test the actual standalone transport framing and telemetry parser."""
import struct
import unittest
from deployment.runtime.rrc_serial import packet, motor_packet, PacketParser

class RRCSerialTests(unittest.TestCase):
    def test_fragmented_stream_and_sensor_fields(self):
        parser=PacketParser()
        imu=packet(7,struct.pack('<6f',0,0,1,0,0,0))
        battery=packet(0,b'\x04'+struct.pack('<H',7400))
        result=[]
        for byte in b'noise'+imu+battery:
            result.extend(parser.feed(bytes([byte])))
        self.assertEqual(result,[(7,struct.pack('<6f',0,0,1,0,0,0)),(0,b'\x04'+struct.pack('<H',7400))])
        self.assertEqual(parser.valid,2)

    def test_crc_failure_recovers_and_motor_packet_matches_verified_bytes(self):
        parser=PacketParser()
        good=packet(0,b'\x04\xe8\x1c')
        bad=good[:-1]+bytes([good[-1]^1])
        self.assertEqual(parser.feed(bad+good),[(0,b'\x04\xe8\x1c')])
        self.assertEqual(parser.bad_crc,1)
        self.assertEqual(motor_packet([.6,0,0,0]).hex(),'aa5503160104009a99193f0100000000020000000003000000008d')
        with self.assertRaises(ValueError):motor_packet([1,2,3,float('inf')])
