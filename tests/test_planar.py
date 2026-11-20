"""Real polygon geometry, Gym semantics and shared-core adapter parity."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from mentorpi_nav.core import NavigationCfg,NavigationState
from mentorpi_nav.geometry import (rectangle,polygon_distance,polygon_raycast,polygon_sweep,
                                   point_polygon_distance,segment_polygon_clear)
from mentorpi_nav.planar_env import PlanarBatch,PlanarGymEnv,RslPlanarEnv
from mentorpi_nav.teacher import PathTeacher,line_clear


def config(name='lidar_box_doorway'):
    return NavigationCfg(**json.loads((Path(__file__).resolve().parents[1]/'scenarios'/f'{name}.json').read_text()))


class GeometryTests(unittest.TestCase):
    def tensors(self,poly):
        return torch.tensor([[poly]],dtype=torch.float64),torch.tensor([[len(poly)]])

    def test_rectangle_flat_face_and_concave_pocket(self):
        vertices,counts=self.tensors(rectangle([1,0],[.4,1.]))
        rays=torch.tensor([[[1.,0.],[1.,.1]]],dtype=torch.float64)
        rays=rays/rays.norm(dim=-1,keepdim=True)
        ranges=polygon_raycast(torch.zeros(1,2,dtype=torch.float64),rays,vertices,counts)
        np.testing.assert_allclose((ranges[...,None]*rays)[0,:,0],[.8,.8],atol=1e-9)
        poly=[[0.,0.],[1.,0.],[1.,.3],[.3,.3],[.3,1.],[0.,1.]]
        vertices,counts=self.tensors(poly)
        points=torch.tensor([[[.6,.6],[.1,.6],[1.1,.1]]],dtype=torch.float64)
        distances=polygon_distance(points,vertices,counts)[0].tolist()
        self.assertGreater(distances[0],0.)
        self.assertLess(distances[1],0.)
        for p,d in zip(points[0].tolist(),distances):
            self.assertAlmostEqual(point_polygon_distance(p,poly),d,places=8)

    def test_sweep_prevents_tunneling_rotated_edges_and_matches_teacher(self):
        poly=rectangle([0,0],[.1,1.],math.pi/6)
        vertices,counts=self.tensors(poly)
        for start,end,expected in [([-1,0],[1,0],True),([-1,.9],[1,.9],False),([0,0],[0,0],True)]:
            actual=bool(polygon_sweep(torch.tensor([start],dtype=torch.float64),torch.tensor([end],dtype=torch.float64),vertices,counts,.15)[0])
            self.assertEqual(actual,expected)
            self.assertEqual(segment_polygon_clear(start,end,poly,.15),not expected)

    def test_empty_slots_and_partial_reset_geometry(self):
        state=NavigationState(2,config(),seed=4);state.reset()
        other=state.polygons[1].clone()
        state.polygons[0]=99
        state.reset([0])
        torch.testing.assert_close(state.polygons[0],other)
        torch.testing.assert_close(state.polygons[1],other)
        self.assertEqual(int((state.polygon_counts[0]>0).sum()),4)

    def test_scene_sensor_and_teacher_use_the_box_outline(self):
        state=NavigationState(1,replace(config(),lidar_noise_std=0.,lidar_dropout=0.));state.reset()
        state.observations()
        self.assertAlmostEqual(float(state.lidar.ranges[0,0]),.387242,places=5)
        teacher=PathTeacher(state)
        self.assertIsNotNone(teacher.paths[0])
        self.assertGreater(len(teacher.paths[0]),2)
        for a,b in zip(teacher.paths[0],teacher.paths[0][1:]):
            self.assertTrue(line_clear(a,b,teacher.obstacles[0],state.cfg.robot_radius,state.cfg.arena_half_extent,teacher.polygons[0]))

    def test_room_layouts_vary_and_have_reachable_routes(self):
        state=NavigationState(4,config('lidar_rooms'),seed=77);state.reset()
        self.assertFalse(torch.equal(state.polygons[0],state.polygons[1]))
        teacher=PathTeacher(state)
        self.assertTrue(all(p is not None for p in teacher.paths))
        self.assertTrue((state.polygon_counts==6).any())


class AdapterTests(unittest.TestCase):
    def test_clean_scenario_clears_inherited_lidar_and_polygon_scene(self):
        root=Path(__file__).resolve().parents[1]/'scenarios'
        clean=json.loads((root/'original_four.json').read_text())
        inherited=replace(config('lidar_rooms'),**clean)
        self.assertIsNone(inherited.scene_spec)
        self.assertEqual(inherited.occupancy_source,'geometry')
        self.assertEqual(inherited.drive_model,'body')
        a=NavigationState(2,inherited,seed=456);a.reset()
        b=NavigationState(2,replace(NavigationCfg(),**clean),seed=456);b.reset()
        for key in a.observations():torch.testing.assert_close(a.observations()[key],b.observations()[key])

    def test_batch_matches_core_motion_observations_and_rewards(self):
        cfg=config();batch=PlanarBatch(cfg,2,seed=21)
        core=NavigationState(2,cfg,seed=21);core.reset()
        core.observations()  # Both adapters acquire the initial scan at reset.
        for _ in range(8):
            action=torch.tensor([[.3,.1,0.],[-.1,0.,.2]])
            core.step(action);obs,reward,term,trunc,info=batch.step(action)
            torch.testing.assert_close(batch.nav.xy,core.xy)
            torch.testing.assert_close(reward,core.reward)
            expected=core.observations()
            for k in expected:torch.testing.assert_close(obs[k],expected[k])

    def test_timeout_and_collision_are_distinct_and_autoreset_keeps_outcomes(self):
        cfg=replace(config(),episode_seconds=.05)
        env=RslPlanarEnv(cfg,2)
        _,_,done,info=env.step(torch.zeros(2,3))
        self.assertTrue(done.bool().all());self.assertTrue(info['time_outs'].all())
        self.assertTrue((env.episode_length_buf==0).all())
        env=RslPlanarEnv(config(),1)
        env.batch.nav.xy[0]=torch.tensor([-1.,0.])
        _,_,done,info=env.step(torch.zeros(1,3))
        self.assertTrue(bool(done[0]));self.assertFalse(bool(info['time_outs'][0]))
        self.assertTrue(bool(info['outcomes']['collision'][0]))
        self.assertFalse(bool(env.batch.nav.collision[0]))

    def test_gym_reset_seed_and_spaces(self):
        from gymnasium.utils.env_checker import check_env
        env=PlanarGymEnv(config())
        check_env(env,skip_render_check=True)
        obs,_=env.reset(seed=8)
        self.assertTrue(env.observation_space.contains(obs))

    def test_autoreset_does_not_sample_a_discarded_terminal_scan(self):
        cfg=config();env=RslPlanarEnv(cfg,1,seed=91)
        reference=NavigationState(1,cfg,seed=91);reference.reset();reference.observations()
        resets=0
        for _ in range(70):
            action=torch.tensor([[1.,0.,0.]])
            reference.step(action)
            expected_reward=reference.reward.clone()
            done=reference.success|reference.collision
            if done.any():
                reference.reset(done.nonzero().flatten());resets+=1
            obs,reward,actual_done,_=env.step(action)
            torch.testing.assert_close(reward,expected_reward)
            torch.testing.assert_close(actual_done.bool(),done)
            expected=reference.observations()
            for key in expected:torch.testing.assert_close(obs[key],expected[key])
        self.assertGreater(resets,0)


if __name__=='__main__':unittest.main()
