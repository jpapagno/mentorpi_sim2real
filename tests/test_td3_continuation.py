from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from mentorpi_nav.core import NavigationCfg,NavigationState
from mentorpi_nav.td3 import TD3


class TD3ContinuationTests(unittest.TestCase):
    def test_unexpected_error_preserves_checkpoint_and_propagates(self):
        from mentorpi_nav.cli.td3 import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'new'
            cfg=NavigationCfg(obstacle_count=0,policy_input='laser_sectors',occupancy_source='lidar',translation_only=True)
            scenario=root/'scenario.json';scenario.write_text(json.dumps(asdict(cfg)))
            argv=['td3','--scenario-file',str(scenario),'--output',str(out),'--device','cpu','--num-envs','4',
                  '--transitions','4','--random-steps','0','--batch-size','4','--replay-size','32']
            with patch.object(sys,'argv',argv),patch.object(TD3,'update',side_effect=RuntimeError('injected failure')):
                with self.assertRaisesRegex(RuntimeError,'injected failure'):main()
            saved=torch.load(out/'crash_4.pt',weights_only=True)
            self.assertTrue(saved['translation_only'])
            self.assertIn('critic_optimizer',saved)

    def test_forever_ignores_transition_limit_and_interrupt_saves(self):
        from mentorpi_nav.cli.td3 import main
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();out=root/'new'
            cfg=NavigationCfg(obstacle_count=0,policy_input='laser_sectors',occupancy_source='lidar',translation_only=True)
            scenario=root/'scenario.json';scenario_values=asdict(cfg);scenario_values.pop('speed_limits')
            scenario.write_text(json.dumps(scenario_values))
            (source/'task.json').write_text(json.dumps({'network_architecture':'laser_sector_td3_v1','navigation':asdict(cfg)}))
            learner=TD3(translation_only=True);learner.updates=20;learner.save(source/'model.pt',100)
            calls=[]
            def interrupt_after_updates(self,batch):
                calls.append(1)
                if len(calls)>=6:raise KeyboardInterrupt
                return {}
            argv=['td3','--scenario-file',str(scenario),'--output',str(out),'--device','cpu','--num-envs','4',
                  '--transitions','4','--forever','--random-steps','0','--batch-size','4','--replay-size','32',
                  '--save-every','4','--keep-checkpoints','2','--init-checkpoint',str(source/'model.pt')]
            with patch.object(sys,'argv',argv),patch.object(TD3,'update',interrupt_after_updates):main()
            saved=sorted(out.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
            self.assertGreater(int(saved[-1].stem.split('_')[-1]),4)
            self.assertLessEqual(len(saved),3) # periodic retention plus interruption save
            checkpoint=torch.load(saved[-1],weights_only=True)
            self.assertEqual(checkpoint['updates'],20)
            for key,value in learner.actor.state_dict().items():torch.testing.assert_close(checkpoint['actor_state_dict'][key],value)
            task=json.loads((out/'task.json').read_text());self.assertEqual(task['source_transitions'],100)
            self.assertTrue(task['fresh_replay'])

    def test_mixture_samples_many_seeds_without_invalid_layouts(self):
        import random
        from mentorpi_nav.scenes import sample_scene
        config=NavigationCfg(**json.loads(Path('scenarios/lidar_translation_varied_forever.json').read_text()))
        rng=random.Random(73010)
        for _ in range(100):
            start,goal,circles,yaw,polygons=sample_scene(config,rng)
            self.assertGreaterEqual(len(start),2)
            self.assertLessEqual(len(polygons),config.max_polygons)
