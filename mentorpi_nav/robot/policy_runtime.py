"""Prepare fixed-shape TorchScript inference before any motion can be enabled."""
import time
import numpy as np
import torch

def warm_actor(actor, warmup=25, samples=10):
    state=torch.zeros((1,8),dtype=torch.float32)
    grid=torch.ones((1,1,100,100),dtype=torch.float32)
    timings=[]
    with torch.no_grad():
        for index in range(warmup+samples):
            start=time.perf_counter()
            output=actor(state,grid)
            elapsed=(time.perf_counter()-start)*1000
            if output.shape!=(1,3) or not torch.isfinite(output).all():
                raise ValueError('Policy warmup returned invalid action')
            if index>=warmup:timings.append(elapsed)
    result={'warmup_calls':warmup,'measured_calls':samples,
            'inference_ms':{str(p):float(np.percentile(timings,p)) for p in (50,95,100)}}
    if result['inference_ms']['100']>100:
        raise RuntimeError('Warmed actor exceeds 100 ms control budget: '+str(result))
    return result
