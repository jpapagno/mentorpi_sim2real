from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
import numpy as np
import torch

from mentorpi_nav.core import NavigationCfg,NavigationState
from mentorpi_nav.robot.laser_sectors import sector_observation,sector_observation_torch,OBS_SIZE
from mentorpi_nav.td3 import TD3,ReplayBuffer


class SectorContractTests(unittest.TestCase):
    def test_numpy_torch_parity_and_cardinal_axes(self):
        points=np.array([[1,0],[.4,0],[0,1],[0,-1],[-1,0],[np.inf,0],[np.nan,0]],dtype=np.float64)
        expected=sector_observation(points,[.6,.3],[.1,-.2,.3])
        actual=sector_observation_torch(torch.tensor(points)[None],torch.tensor([[.6,.3]],dtype=torch.float64),torch.tensor([[.1,-.2,.3]]))
        np.testing.assert_allclose(actual[0].numpy(),expected,atol=1e-7)
        self.assertAlmostEqual(expected[18],.4/2.5,places=6)
        self.assertAlmostEqual(expected[27],1/2.5,places=6)
        self.assertAlmostEqual(expected[9],1/2.5,places=6)
        self.assertAlmostEqual(expected[0],1/2.5,places=6)
        self.assertEqual(expected[1],0)

    def test_missing_and_range_clipping(self):
        obs=sector_observation([[10,0]],[0,0],[0,0,0])
        self.assertEqual(obs.shape,(OBS_SIZE,));self.assertEqual(obs[18],1)
        self.assertTrue((sector_observation([], [0,0],[0,0,0])[:36]==0).all())

    def test_lazy_grid_and_scan_cadence_parity(self):
        path=Path(__file__).resolve().parents[1]/'scenarios/lidar_sector_td3.json'
        cfg=NavigationCfg(**json.loads(path.read_text()))
        a=NavigationState(2,cfg,seed=8123);a.reset()
        b=NavigationState(2,replace(cfg,policy_input='occupancy'),seed=8123);b.reset()
        for _ in range(3):
            self.assertEqual(a.observations()['policy'].shape,(2,41))
            torch.testing.assert_close(a.occupancy_grid(),b.occupancy_grid())
            torch.testing.assert_close(a.lidar.ranges,b.lidar.ranges)
            action=torch.zeros(2,3);a.step(action);b.step(action)
        a.reset();b.reset()
        a.observations();torch.testing.assert_close(a.occupancy_grid(),b.occupancy_grid())

    def test_mecanum_reward_toward_goal_and_terminal(self):
        path=Path(__file__).resolve().parents[1]/'scenarios/lidar_sector_td3.json'
        cfg=NavigationCfg(**json.loads(path.read_text()))
        cfg=replace(cfg,scene_spec={'layout':'fixed','start':[-.3,0],'goal':[.3,0],'obstacles':[]},min_goal_distance=.5,drive_model='body',action_delay_steps=0,actuator_scale_range=None)
        toward=NavigationState(1,cfg);toward.reset();toward.step(torch.tensor([[1.,0.,0.]]))
        away=NavigationState(1,cfg);away.reset();away.step(torch.tensor([[-1.,0.,0.]]))
        self.assertGreater(float(toward.reward[0]),float(away.reward[0]))
        toward.xy[:]=toward.goal; toward.velocity[:]=0;toward.step(torch.zeros(1,3))
        self.assertTrue(toward.success[0]);self.assertEqual(float(toward.reward[0]),100.)
        away.xy[:]=torch.tensor([[.84,0]]);away.step(torch.zeros(1,3))
        self.assertTrue(away.collision[0]);self.assertEqual(float(away.reward[0]),-100.)


class TD3Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(1)

    def test_sector_export_and_runtime_warmup(self):
        from mentorpi_nav.robot.policy_runtime import warm_actor
        learner=TD3()
        exported=torch.jit.script(learner.actor)
        observation=torch.randn(5,41)
        torch.testing.assert_close(exported(observation),learner.actor(observation))
        report=warm_actor(exported,warmup=2,samples=2,sector_input=True)
        self.assertEqual(report['measured_calls'],2)

    def test_terminal_bootstrap_and_delayed_target_update(self):
        learner=TD3(gamma=.5,lr=0)
        for net in (learner.critic,learner.critic_target):
            for p in net.parameters():p.data.zero_()
            net.q1[-1].bias.data.fill_(1);net.q2[-1].bias.data.fill_(1)
        batch=[torch.zeros(2,41),torch.zeros(2,3),torch.full((2,1),2.),torch.zeros(2,41),torch.tensor([[0.],[1.]])]
        # Timeout/nonterminal target=2+.5*1; true terminal target=2.
        metrics=learner.update(batch)
        self.assertAlmostEqual(metrics['critic_loss'],3.25,places=6)
        self.assertNotIn('actor_loss',metrics)
        self.assertIn('actor_loss',learner.update(batch))

    def test_updates_and_action_bounds(self):
        learner=TD3();batch=[torch.randn(16,41),torch.rand(16,3)*2-1,torch.randn(16,1),torch.randn(16,41),torch.zeros(16,1)]
        actor_before=[p.detach().clone() for p in learner.actor.parameters()]
        learner.update(batch)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actor_before,learner.actor.parameters())))
        learner.update(batch)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(actor_before,learner.actor.parameters())))
        self.assertLessEqual(float(learner.actor(torch.randn(8,41)).detach().abs().max()),1)

    def test_random_runtime_observation_parity(self):
        rng=np.random.default_rng(721)
        points=rng.normal(size=(4,720,2))
        points[:,::11]=np.nan
        goals=rng.normal(size=(4,2));previous=rng.uniform(-1,1,(4,3))
        actual=sector_observation_torch(torch.tensor(points),torch.tensor(goals),torch.tensor(previous)).numpy()
        expected=np.stack([sector_observation(p,g,a) for p,g,a in zip(points,goals,previous)])
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)

    def test_replay_wrap_and_final_observation_copy(self):
        replay=ReplayBuffer(4)
        state=torch.zeros(3,41);next_state=torch.ones(3,41)
        replay.add(state,torch.zeros(3,3),torch.ones(3),next_state,torch.tensor([False,True,False]))
        next_state.zero_();self.assertTrue((replay.next_state[:3]==1).all())
        replay.add(state,torch.zeros(3,3),torch.ones(3),state,torch.zeros(3))
        self.assertEqual(replay.size,4);self.assertEqual(replay.position,2)
        self.assertEqual(replay.sample(2,'cpu')[0].shape,(2,41))
