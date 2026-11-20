"""Prepare fixed-shape TorchScript inference before any motion can be enabled."""
import time
import numpy as np
import torch

def warm_actor(actor, warmup=50, samples=10, sector_input=False):
    # Match the NumPy views returned by NavigationIO, including zero strides
    # on singleton batch/channel axes. A contiguous all-zero Torch fixture can
    # leave the first real-input TorchScript specialization cold.
    state_values=np.zeros(8,dtype=np.float32)
    state_values[0]=1/(4*np.sqrt(2))
    state=torch.from_numpy(state_values[None,:])
    grid_values=np.ones((100,100),dtype=np.float32)
    grid_values[47:53,47:53]=0
    grid=torch.from_numpy(grid_values[None,None,:,:])
    if sector_input:
        from .laser_sectors import OBS_SIZE,GOAL_SCALE
        values=np.zeros(OBS_SIZE,dtype=np.float32);values[36]=1/GOAL_SCALE
        state=torch.from_numpy(values[None,:])
    timings=[]
    with torch.no_grad():
        for index in range(warmup+samples):
            start=time.perf_counter()
            output=actor(state) if sector_input else actor(state,grid)
            elapsed=(time.perf_counter()-start)*1000
            if output.shape!=(1,3) or not torch.isfinite(output).all():
                raise ValueError('Policy warmup returned invalid action')
            if index>=warmup:timings.append(elapsed)
    result={'warmup_calls':warmup,'measured_calls':samples,
            'input_layout':'runtime_numpy_views',
            'inference_ms':{str(p):float(np.percentile(timings,p)) for p in (50,95,100)}}
    if result['inference_ms']['100']>100:
        raise RuntimeError('Warmed actor exceeds 100 ms control budget: '+str(result))
    return result
