import unittest
import torch
from mentorpi_nav.translation_response import response_rate,translation_response_torch,validate_translation_response,DIRECTIONS,AxisTranslationResponse,BatchedTranslationResponse
from mentorpi_nav.cli.fit_translation import fit
from mentorpi_nav.core import NavigationCfg,NavigationState


class TranslationResponseTests(unittest.TestCase):
    def history(self):
        return self.profile()|{'version':'translation_history_v1','breakaway_distance_m':[.014,.016],'idle_memory_decay_s':10.}

    def test_history_scalar_batch_parity_reset_and_yaw(self):
        profile=self.history();batch=BatchedTranslationResponse(2,'cpu',profile)
        scalar=[AxisTranslationResponse(profile,i) for i in range(2)]
        for command in ([.08,-.1,.2],)*8+([0,0,0],)*20+([-.08,.1,-.2],)*15:
            actual=batch.step(torch.tensor([command,command]),.05)
            for axis in (0,1):self.assertAlmostEqual(float(actual[0,axis]),scalar[axis].step(command[axis],.05),places=6)
            self.assertAlmostEqual(float(actual[0,2]),command[2],places=6)
        other=batch.memory[1].clone();batch.reset(torch.tensor([0]))
        self.assertEqual(float(batch.memory[0].abs().sum()),0)
        torch.testing.assert_close(batch.memory[1],other)

    def test_history_repeat_moves_more_than_reversal(self):
        model=AxisTranslationResponse(self.history(),0)
        def pulse(value):
            travel=sum(model.step(value,.05)*.05 for _ in range(20))
            for _ in range(30):model.step(0,.05)
            return abs(travel)
        first=pulse(.08);repeat=pulse(.08);reverse=pulse(-.08)
        self.assertGreater(repeat,first)
        self.assertGreater(repeat,reverse)

    def profile(self):
        return {'version':'translation_deadzone_v1','directions':{name:{'gain':1.1+i*.1,'deadband_mps':.01+i*.002} for i,name in enumerate(DIRECTIONS)}}

    def test_scalar_batch_signed_parity_and_no_added_yaw(self):
        profile=self.profile();validate_translation_response(profile)
        velocities=torch.tensor([[.08,-.06,.2],[-.1,.001,0],[0,0,-.1]])
        actual=translation_response_torch(velocities,profile)
        for i,row in enumerate(velocities):
            for axis in range(2):self.assertAlmostEqual(float(actual[i,axis]),response_rate(float(row[axis]),axis,profile),places=6)
        torch.testing.assert_close(actual[:,2],velocities[:,2])

    def test_fit_recovers_parameters_without_using_heldout_values(self):
        profile=self.profile();rows=[]
        for run in ('run_01','run_02'):
            for axis in range(2):
                for sign in (1,-1):
                    for speed in (.06,.08,.1):
                        for repeat in range(2):
                            name=DIRECTIONS[2*axis+int(sign<0)]
                            observed=abs(response_rate(sign*speed,axis,profile))
                            rows.append({'run':run,'direction':name,'measured_m':observed,'cross_axis_m':0,
                                         'gyro_angle_rad':0,'intervals':[(speed,1)]})
        recovered,_,validation=fit(rows,['run_01'])
        self.assertLess(validation['max_abs_error_m'],1e-5)
        for name in DIRECTIONS:
            self.assertAlmostEqual(recovered['directions'][name]['gain'],profile['directions'][name]['gain'],places=3)
        changed=[r|{'measured_m':r['measured_m']+.03} if r['run']=='run_02' else r for r in rows]
        second,_,bad=fit(changed,['run_01'])
        self.assertEqual(second,recovered)
        self.assertGreater(bad['rmse_m'],.029)

    def test_shared_core_applies_response_without_altering_command(self):
        profile=self.profile()
        state=NavigationState(1,NavigationCfg(obstacle_count=0,translation_only=True,translation_response=profile))
        state.goal[:]=1
        state.step(torch.tensor([[.5,0,1.]]))
        self.assertAlmostEqual(float(state.velocity[0,0]),response_rate(.025,0,profile),places=6)
        self.assertEqual(float(state.velocity[0,2]),0)

    def test_bad_profile_rejected(self):
        profile=self.profile();profile['directions']['left']['deadband_mps']=.2
        with self.assertRaises(ValueError):NavigationCfg(translation_response=profile)
