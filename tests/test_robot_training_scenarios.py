import random
import unittest
import torch
from mentorpi_nav.core import NavigationCfg,NavigationState,sample_layout


class RobotTrainingScenarioTests(unittest.TestCase):
    def test_command_delay_and_partial_reset_isolation(self):
        state=NavigationState(2,NavigationCfg(obstacle_count=0,action_delay_steps=2))
        state.xy.zero_();state.yaw.zero_();state.goal[:]=torch.tensor([1.,0.])
        state.step(torch.tensor([[1.,0.,0.],[1.,0.,0.]]))
        self.assertTrue((state.xy==0).all())
        state.reset([0]);state.xy[0]=0;state.yaw[0]=0
        state.step(torch.zeros(2,3))
        self.assertTrue((state.xy==0).all())
        state.step(torch.zeros(2,3))
        self.assertEqual(float(state.xy[0,0]),0.)
        self.assertGreater(float(state.xy[1,0]),0.)

    def test_compound_shapes_and_actuator_scale(self):
        cfg=NavigationCfg(max_obstacles=24,obstacle_count=24,min_obstacles=10,layout_mode='shapes',grid_resolution=.1)
        _,_,obstacles,_=sample_layout(cfg,random.Random(87001))
        self.assertTrue(10<=len(obstacles)<=24)
        touching=any((a[0]-b[0])**2+(a[1]-b[1])**2<(a[2]+b[2])**2
                     for i,a in enumerate(obstacles) for b in obstacles[i+1:])
        self.assertTrue(touching)
        state=NavigationState(1,NavigationCfg(obstacle_count=0,actuator_scale_range=(.6,.6)))
        state.reset();state.xy.zero_();state.yaw.zero_();state.goal[:]=torch.tensor([1.,0.])
        for _ in range(20):state.step(torch.tensor([[1.,0.,0.]]))
        self.assertAlmostEqual(float(state.velocity[0,0]),.12,places=6)
