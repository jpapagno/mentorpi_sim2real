import json
from pathlib import Path
import tempfile
import unittest
from mentorpi_nav.robot.calibration_suite import protocol,summarize


class SuiteTests(unittest.TestCase):
    def test_translation_protocol_never_requests_yaw(self):
        plan=protocol(True)
        self.assertEqual([b['linear'] for b in plan],[.1,.08,.06])
        self.assertEqual(sum(len(b['directions']) for b in plan),24)
        self.assertTrue(all(d not in ('cw','ccw') for b in plan for d in b['directions']))
    def test_balanced_bounded_protocol(self):
        opposites={'forward':'backward','left':'right','ccw':'cw'}
        for block in protocol():
            self.assertLessEqual(block['linear'],.1)
            self.assertLessEqual(block['yaw'],.3)
            for a,b in opposites.items():
                self.assertEqual(block['directions'].count(a),block['directions'].count(b))

    def test_matching_slots_and_aborted_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for i,value in enumerate([.05,.07]):
                run=Path(directory)/f'run_{i}';run.mkdir()
                pulse={'direction':'forward','body_delta':[value,0,0],'laser_body_delta':[value,0,0],
                       'settling_body_delta':[0,0,0],'gyro_angle_radians':0,'integrated_applied_velocity':[.1,0,0],'axis_gain':value/.1}
                (run/'block_00.json').write_text(json.dumps({'passed':i==0,'pulses':[pulse],'error':'abort' if i else None}))
            report=summarize(directory)
            metric=report['groups']['block_00/0/forward']['body_0']
            self.assertEqual(metric['n'],2)
            self.assertAlmostEqual(metric['mean'],.06)
            self.assertGreater(metric['std'],0)
            self.assertEqual(len(report['failures']),1)
