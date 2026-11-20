from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
from mentorpi_nav.core import NavigationCfg
from mentorpi_nav.scenes import sample_scene


class SceneSamplingRetryTests(unittest.TestCase):
    def test_transient_exhaustion_retries_same_scene_family(self):
        cfg=NavigationCfg(**json.loads(Path('scenarios/lidar_translation_varied_forever.json').read_text()))
        empty={'layout':'clutter','rectangle_count_range':[0,0],'rectangle_size_range':[.12,.22]}
        cfg=replace(cfg,min_goal_distance=.001,scene_spec={'layout':'mixture','choices':[{'name':'open','scene':empty}]})
        calls=[]
        def planner(*args,**kwargs):
            calls.append(1)
            return None if len(calls)<=100 else [args[0],args[1]]
        with patch('mentorpi_nav.teacher.plan_path',side_effect=planner):
            result=sample_scene(cfg,random.Random(73010))
        self.assertGreater(len(calls),100)
        self.assertEqual(result[2],[])

    def test_invalid_configuration_is_not_swallowed(self):
        cfg=NavigationCfg(scene_spec={'layout':'mixture','choices':[{'scene':{'layout':'invalid'}}]})
        with self.assertRaisesRegex(ValueError,'Unknown scene layout'):sample_scene(cfg,random.Random(1))
