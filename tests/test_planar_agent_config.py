import unittest
from mentorpi_nav.model_spec import planar_ppo_config


class PlanarAgentConfigTests(unittest.TestCase):
    def test_default_entropy_preserved(self):
        self.assertEqual(planar_ppo_config()['algorithm']['entropy_coef'],.001)

    def test_entropy_override_is_only_changed_setting(self):
        baseline=planar_ppo_config()
        changed=planar_ppo_config(entropy_coef=.01)
        changed['algorithm']['entropy_coef']=baseline['algorithm']['entropy_coef']
        self.assertEqual(changed,baseline)

    def test_invalid_entropy(self):
        for value in [-.01,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                planar_ppo_config(entropy_coef=value)
