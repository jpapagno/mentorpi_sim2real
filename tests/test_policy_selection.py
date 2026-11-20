import json
from pathlib import Path
import tempfile
import unittest
from mentorpi_nav.robot.policy_selection import resolve_policy


class PolicySelectionTests(unittest.TestCase):
    def test_explicit_selection_preserves_active_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=root/'models';candidate=models/'candidate';candidate.mkdir(parents=True)
            for folder in (models,candidate):
                (folder/'policy.ts').write_text(folder.name)
                (folder/'contract.json').write_text('{}')
            (models/'catalog.json').write_text(json.dumps({'policies':{'2m':'candidate','bad':'../../outside'}}))
            self.assertEqual(resolve_policy(root,'2m'),candidate.resolve())
            self.assertEqual(resolve_policy(root),models.resolve())
            self.assertEqual((models/'policy.ts').read_text(),'models')
            with self.assertRaises(ValueError):resolve_policy(root,'bad')
            with self.assertRaises(ValueError):resolve_policy(root,'missing')
