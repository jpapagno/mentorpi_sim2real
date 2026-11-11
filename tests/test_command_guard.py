import unittest
from mentorpi_nav.robot.command_guard import CommandGuard


class CommandGuardTests(unittest.TestCase):
    def healthy(self):
        gate=CommandGuard()
        gate.enabled=True
        gate.enable_at=gate.scan_at=gate.imu_at=10.
        gate.clearance=1.
        gate.receive_command((.3,-.3,.8),10.)
        return gate

    def test_default_disarmed_and_bounded_output(self):
        self.assertEqual(CommandGuard().output(10.)[0],(0.,0.,0.))
        self.assertEqual(self.healthy().output(10.1)[0],(.1,-.1,.3))

    def test_independent_staleness_stops(self):
        for field,reason in [('command_at','command_stale'),('enable_at','disarmed'),
                             ('scan_at','scan_stale'),('imu_at','imu_stale')]:
            gate=self.healthy()
            setattr(gate,field,9.)
            output,reasons=gate.output(10.1)
            self.assertEqual(output,(0.,0.,0.))
            self.assertIn(reason,reasons)

    def test_clearance_is_diagnostic_and_nonfinite_input_stops(self):
        gate=self.healthy();gate.clearance=.21
        self.assertEqual(gate.output(10.1)[0],(.1,-.1,.3))
        gate=self.healthy();gate.receive_command((float('nan'),0.,0.),10.)
        self.assertIn('command_stale',gate.output(10.1)[1])

    def test_arming_never_replays_a_disarmed_command(self):
        gate=self.healthy()
        gate.receive_enable(False,10.)
        gate.receive_command((.1,0.,0.),10.01)
        gate.receive_enable(True,10.02)
        self.assertEqual(gate.output(10.03)[0],(0.,0.,0.))
        gate.receive_command((.1,0.,0.),10.04)
        self.assertEqual(gate.output(10.05)[0],(.1,0.,0.))
