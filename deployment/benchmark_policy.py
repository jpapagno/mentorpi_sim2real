"""Stationary onboard numerical parity and single-environment inference timing."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root=Path('/home/ubuntu/shared/mentorpi_nav')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir',default=str(root/'models'))
parser.add_argument('--output-name',default='inference.json')
args=parser.parse_args()
model_dir=Path(args.model_dir)
cases=np.load(model_dir/'parity_cases.npz')
model=torch.jit.load(str(model_dir/'policy.ts'),map_location='cpu').eval()
states=torch.from_numpy(cases['policy'])
grids=torch.from_numpy(cases['occupancy'])
with torch.no_grad():
    actual=model(states,grids).numpy()
    error=float(np.max(np.abs(actual-cases['action'])))
    np.testing.assert_allclose(actual,cases['action'],rtol=2e-4,atol=2e-5)
    for _ in range(20):model(states[:1],grids[:1])
    timings=[]
    for _ in range(300):
        start=time.perf_counter()
        model(states[:1],grids[:1])
        timings.append((time.perf_counter()-start)*1000)
result={'passed':True,'runtime':'torchscript','torch_version':torch.__version__,
        'numpy_version':np.__version__,'max_action_error':error,
        'inference_ms':{str(p):float(np.percentile(timings,p)) for p in [50,95,99,100]}}
(root/'diagnostics'/args.output_name).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
