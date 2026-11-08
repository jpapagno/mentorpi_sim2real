"""Select the parity-verified model while the service is stopped/disarmed."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path('/home/pi/docker/tmp/mentorpi_nav')
candidate=root/'models/shape_499'
parity=json.loads((root/'diagnostics/inference_shape_499.json').read_text())
if not parity.get('passed') or parity['max_action_error']>2e-5:
    raise RuntimeError('Candidate has no acceptable onboard parity evidence')
subprocess.run(['sudo','-n','systemctl','stop','mentorpi-navigation.service'],check=True)
for filename in ['policy.ts','policy.onnx','contract.json','parity_cases.npz']:
    destination=root/'models'/filename
    temporary=destination.with_suffix(destination.suffix+'.next')
    shutil.copyfile(candidate/filename,temporary)
    temporary.replace(destination)
selection={'variant':'shape_499','source_checkpoint':'logs/robot_shape_ppo/model_499.pt',
    'lookahead_m':.5,'torchscript_sha256':hashlib.sha256((root/'models/policy.ts').read_bytes()).hexdigest(),
    'physical_acceptance':'pending_motor_commissioning','onboard_parity':parity,
    'offline_stack_success_rates':{'456':.98,'789':.995,'2026':.995}}
(root/'models/selection.json').write_text(json.dumps(selection,indent=2))
subprocess.run(['sudo','-n','systemctl','start','mentorpi-navigation.service'],check=True)
print(json.dumps(selection,indent=2),flush=True)
