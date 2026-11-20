import tempfile
from pathlib import Path
import unittest
import torch
from mentorpi_nav.core import NavigationCfg,NavigationState
from mentorpi_nav.td3 import SectorActor,TD3


class TranslationOnlyTests(unittest.TestCase):
    def test_actor_export_and_checkpoint_keep_zero_yaw(self):
        actor=SectorActor(True)
        obs=torch.randn(17,41)
        output=actor(obs)
        self.assertTrue(torch.equal(output[:,2],torch.zeros(17)))
        self.assertGreater(float(output[:,:2].abs().sum()),0)
        scripted=torch.jit.script(actor)
        torch.testing.assert_close(scripted(obs),output)
        with tempfile.TemporaryDirectory() as directory:
            filename=Path(directory)/'model.pt'
            TD3(translation_only=True).save(filename,0)
            self.assertTrue(torch.load(filename,weights_only=True)['translation_only'])

    def test_environment_masks_commands_before_delay_and_history(self):
        cfg=NavigationCfg(translation_only=True,obstacle_count=0,action_delay_steps=2,drive_model='mecanum_v1')
        state=NavigationState(2,cfg);state.reset()
        yaw=state.yaw.clone()
        for _ in range(10):state.step(torch.tensor([[.2,.3,1.],[.3,.2,-1.]]))
        self.assertEqual(float(state.actions[:,2].abs().max()),0)
        self.assertEqual(float(state.delayed_actions[:,:,2].abs().max()),0)
        torch.testing.assert_close(state.yaw,yaw)

    def test_target_smoothing_does_not_reintroduce_yaw(self):
        learner=TD3(translation_only=True)
        actions=[]
        hook=learner.critic_target.register_forward_pre_hook(lambda m,args:actions.append(args[1].clone()))
        learner.update([torch.randn(8,41),torch.zeros(8,3),torch.zeros(8,1),torch.randn(8,41),torch.zeros(8,1)])
        hook.remove()
        self.assertEqual(float(actions[0][:,2].abs().max()),0)
