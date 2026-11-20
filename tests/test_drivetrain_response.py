import unittest
from mentorpi_nav.drivetrain_response import YawResponse,YawResponseCfg


class YawResponseTests(unittest.TestCase):
    def test_batched_replay_matches_scalar_and_resets_independently(self):
        import torch
        from mentorpi_nav.drivetrain_response import BatchedYawResponse
        batched=BatchedYawResponse(2,'cpu');models=[YawResponse(),YawResponse()]
        for i in range(400):
            commands=[.3 if i<150 else (-.3 if i>250 else 0.),-.3 if i<150 else 0.]
            actual=batched.step(torch.tensor(commands),.01)
            for k in range(2):self.assertAlmostEqual(float(actual[k]),models[k].step(commands[k],.01),places=5)
        old=float(batched.memory[1]);batched.reset([0])
        self.assertEqual(float(batched.memory[0]),0);self.assertEqual(float(batched.memory[1]),old)

    def test_shared_core_preserves_command_cap_and_motor_memory(self):
        import torch
        from mentorpi_nav.core import NavigationCfg,NavigationState
        cfg=NavigationCfg(obstacle_count=0,command_limits=(.1,.1,.3),yaw_response={})
        state=NavigationState(1,cfg);state.reset()
        for _ in range(30):state.step(torch.tensor([[0.,0.,1.]]))
        self.assertLessEqual(float(state.body_command_velocity[0,2]),.300001)
        self.assertGreater(float(state.velocity[0,2]),.1)
        state.reset();self.assertEqual(float(state.yaw_response.memory[0]),0)
    def pulse(self,model,command):
        before=model.angle
        for _ in range(150):model.step(command,.01)
        for _ in range(100):model.step(0.,.01)
        return model.angle-before

    def test_history_changes_same_command_response(self):
        model=YawResponse();first=self.pulse(model,.3);same=self.pulse(model,.3)
        self.assertGreater(same,first)
        reversed_angle=abs(self.pulse(model,-.3))
        self.assertLess(reversed_angle,same)

    def test_sign_symmetry_with_mirrored_history(self):
        positive=YawResponse();negative=YawResponse()
        for value in [.3]*150+[0.]*100+[-.3]*150:
            self.assertAlmostEqual(positive.step(value,.01),-negative.step(-value,.01),places=9)

    def test_zero_command_stops_rate_without_erasing_history(self):
        model=YawResponse();self.pulse(model,.3)
        self.assertLess(abs(model.rate),1e-6)
        self.assertGreater(model.memory,0)

    def test_bounds_and_invalid_parameters(self):
        with self.assertRaises(ValueError):YawResponseCfg(rise_time_s=0)
        with self.assertRaises(ValueError):YawResponse().step(.4,.01)
