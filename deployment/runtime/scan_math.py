"""Rebin variable-count full-circle scans without removing sampled obstacles."""
import math
import numpy as np


def fixed_circle_scan(ranges,angle_min,angle_increment,range_min,range_max,size=720):
    source=np.asarray(ranges,dtype=np.float32)
    if len(source)<20 or angle_increment<=0 or abs(len(source)*angle_increment-2*math.pi)>.15:
        raise ValueError('Expected a full-circle scan with valid angular metadata')
    source=np.where(np.isfinite(source)&(source>=range_min)&(source<=range_max),source,np.inf)
    target=-math.pi+np.arange(size)*(2*math.pi/size)
    index=((target-angle_min)%(2*math.pi))/angle_increment
    lower=np.floor(index).astype(int)%len(source)
    upper=(lower+1)%len(source)
    # Minimum of bounding rays preserves thin sampled obstacles conservatively.
    return np.minimum(source[lower],source[upper])
