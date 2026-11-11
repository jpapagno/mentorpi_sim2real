import unittest
import torch
from mentorpi_nav.robot.policy_runtime import warm_actor

class PolicyRuntimeTests(unittest.TestCase):
    def test_warms_real_input_contract_before_timing(self):
        calls=[]
        def actor(state,grid):
            calls.append((tuple(state.shape),tuple(grid.shape)))
            return torch.zeros((1,3))
        result=warm_actor(actor,warmup=3,samples=2)
        self.assertEqual(calls,[((1,8),(1,1,100,100))]*5)
        self.assertEqual(result['measured_calls'],2)

    def test_invalid_actor_never_reaches_ready_state(self):
        with self.assertRaises(ValueError):
            warm_actor(lambda s,g:torch.full((1,3),float('nan')),warmup=1,samples=1)
