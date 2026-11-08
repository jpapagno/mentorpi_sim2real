import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deployment.direct_motor_probe import main,motor_packet,wheel_speeds,ROBOT_SERIAL


class DirectMotorProbeTests(unittest.TestCase):
    def test_packet_matches_captured_vendor_encoding_and_crc(self):
        expected=bytes.fromhex('aa550316010400f2bafabe01f2bafabe02f2bafa3e03f2bafa3e5b')
        self.assertEqual(motor_packet(wheel_speeds('x',.1)),expected)

    def test_unidentified_device_never_opens_a_port(self):
        with patch('deployment.direct_motor_probe.list_ports.comports',return_value=[]), \
             patch.object(sys,'argv',['probe','--pulse','x']), \
             patch('deployment.direct_motor_probe.open_board') as opened:
            with self.assertRaisesRegex(RuntimeError,'not uniquely connected'):main()
            opened.assert_not_called()

    def test_observer_and_bounded_speed_required_before_worker(self):
        port=SimpleNamespace(serial_number=ROBOT_SERIAL,vid=0x1a86,device='TEST_ONLY',description='fixture')
        with patch('deployment.direct_motor_probe.list_ports.comports',return_value=[port]), \
             patch('deployment.direct_motor_probe.subprocess.Popen') as spawned:
            with patch.object(sys,'argv',['probe','--pulse','x']):
                with self.assertRaisesRegex(RuntimeError,'observer recording is required'):main()
            with patch.object(sys,'argv',['probe','--pulse','x','--speed','.2']):
                with self.assertRaisesRegex(ValueError,'exceeds bounded'):main()
            scratch=(Path(__file__).resolve().parents[2]/'.local').resolve()
            with tempfile.TemporaryDirectory(dir=scratch) as folder:
                ready=Path(folder)/'camera.ready.json';ready.write_text('{}')
                os.utime(ready,(time.time()-30,time.time()-30))
                with patch.object(sys,'argv',['probe','--pulse','x','--observer-ready',str(ready)]):
                    with self.assertRaisesRegex(RuntimeError,'not current'):main()
            spawned.assert_not_called()


if __name__=='__main__':unittest.main()
