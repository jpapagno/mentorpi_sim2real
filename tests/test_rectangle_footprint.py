import json
import math
from pathlib import Path
import unittest

import torch

from mentorpi_nav.core import NavigationCfg, NavigationState, world_to_body
from mentorpi_nav.geometry import rectangle_clearance, rectangle_sweep


class RectangleFootprintTests(unittest.TestCase):
    def setUp(self):
        self.xy=torch.zeros(1,2)
        self.yaw=torch.zeros(1)
        self.size=(.254,.2032)
        self.obstacles=torch.zeros(1,1,3)
        self.valid=torch.zeros(1,1,dtype=torch.bool)
        self.polygons=torch.zeros(1,1,8,2)
        self.counts=torch.zeros(1,1,dtype=torch.long)

    def clearance(self):
        return rectangle_clearance(self.xy,self.yaw,self.size,self.obstacles,self.valid,
                                   self.polygons,self.counts,2.).item()

    def test_heading_and_circle_contact(self):
        self.obstacles[0,0]=torch.tensor([0.,.14,.02]);self.valid[:]=True
        self.assertGreater(self.clearance(),0)
        self.yaw[:]=math.pi/2
        self.assertLess(self.clearance(),0)

    def test_crossing_edges_without_contained_vertices(self):
        self.polygons[0,0,:4]=torch.tensor([[-.3,-.01],[.3,-.01],[.3,.01],[-.3,.01]])
        self.counts[:]=4
        self.assertLessEqual(self.clearance(),0)

    def test_rectangle_fits_gap_rejected_by_old_disc(self):
        # 24 cm gap passes a 20.32 cm wide rectangle, but not the old 30 cm disc.
        self.polygons=torch.zeros(1,2,8,2);self.counts=torch.full((1,2),4)
        for i,(lo,hi) in enumerate([(.12,.3),(-.3,-.12)]):
            self.polygons[0,i,:4]=torch.tensor([[-.5,lo],[.5,lo],[.5,hi],[-.5,hi]])
        self.assertAlmostEqual(self.clearance(),.0184,places=5)

    def test_sweep_detects_translation_and_rotation(self):
        self.valid[:]=True;self.obstacles[0,0]=torch.tensor([.4,0.,.01])
        self.assertTrue(rectangle_sweep(self.xy,torch.tensor([[.8,0.]]),self.yaw,self.yaw,
            self.size,self.obstacles,self.valid,self.polygons,self.counts,2.).item())
        # Mid-turn collision; start and 90-degree end poses are clear.
        self.obstacles[0,0]=torch.tensor([.14,.04,.003])
        self.assertGreater(self.clearance(),0)
        self.yaw[:]=math.pi/2;self.assertGreater(self.clearance(),0)
        self.assertTrue(rectangle_sweep(self.xy,self.xy,torch.zeros(1),self.yaw,
            self.size,self.obstacles,self.valid,self.polygons,self.counts,2.).item())

    def test_small_arena_goal_extent_and_normalization(self):
        path=Path(__file__).resolve().parents[1]/'scenarios/lidar_small_rectangle.json'
        cfg=NavigationCfg(**json.loads(path.read_text()))
        self.assertLess(2*math.sqrt(2)*cfg.arena_half_extent,2.475)
        state=NavigationState(8,cfg,seed=71001)
        for _ in range(4):
            state.reset()
            relative=world_to_body(state.goal-state.xy,state.yaw)
            self.assertLess(float(relative.abs().max()),2.475)
            torch.testing.assert_close(state.observations()['policy'][:,:2],relative/(4*math.sqrt(2)))
            self.assertTrue((state.minimum_clearance()>0).all())


if __name__=='__main__':unittest.main()
