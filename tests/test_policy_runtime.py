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

    def test_warmup_matches_live_numpy_strides_and_nonzero_goal(self):
        layouts=[]
        def actor(state,grid):
            layouts.append((state.stride(),grid.stride(),float(state[0,0]),float(grid.min()),float(grid.max())))
            return torch.zeros(1,3)
        result=warm_actor(actor,warmup=3,samples=2)
        for state_stride,grid_stride,goal,minimum,maximum in layouts:
            self.assertEqual(state_stride,(0,1))
            self.assertEqual(grid_stride,(0,0,100,1))
            self.assertGreater(goal,0.)
            self.assertEqual((minimum,maximum),(0.,1.))
        self.assertEqual(result['input_layout'],'runtime_numpy_views')
