"""Robot-aligned occupancy from current observed rays, without a saved map.

Unobserved space stays blocked. Finite LiDAR returns clear their observed ray
and mark its endpoint; removing an object is reflected by the next farther ray.
"""
import numpy as np
from navigation_math import overlay_points

def live_occupancy_from_scan(points,sensor_origin=(0.,0.)):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]<2:raise ValueError('Scan points must be Nx2 or Nx3')
    points=points[np.isfinite(points[:,:2]).all(axis=1),:2]
    origin=np.asarray(sensor_origin,dtype=float)
    if origin.shape!=(2,) or not np.isfinite(origin).all():raise ValueError('Finite sensor origin required')
    visible=np.zeros((100,100),dtype=bool)
    start=np.array([50-origin[0]/.05,50-origin[1]/.05])
    if not ((start>=0)&(start<100)).all():raise ValueError('Sensor origin outside local grid')
    if len(points):
        ends=np.column_stack((50-points[:,0]/.05,50-points[:,1]/.05))
        direction=ends-start
        scale=np.ones(len(points))
        for axis in (0,1):
            positive=direction[:,axis]>0
            negative=direction[:,axis]<0
            limits=np.ones(len(points))
            limits[positive]=(99.999-start[axis])/direction[positive,axis]
            limits[negative]=(0-start[axis])/direction[negative,axis]
            scale=np.minimum(scale,limits)
        direction*=scale[:,None]
        lengths=np.ceil(np.abs(direction).max(axis=1)).astype(int)+1
        steps=np.arange(int(lengths.max())+1)[None,:]
        fraction=steps/lengths[:,None]
        cells=np.floor(start[None,None,:]+direction[:,None,:]*fraction[:,:,None]).astype(int)
        rr,cc=cells[:,:,0],cells[:,:,1]
        valid=(steps<=lengths[:,None])&(rr>=0)&(rr<100)&(cc>=0)&(cc<100)
        visible[rr[valid],cc[valid]]=True
    axis=(49.5-np.arange(100))*.05
    x,y=np.meshgrid(axis,axis,indexing='ij')
    visible|=x*x+y*y<=.15**2
    blocked=~visible
    return overlay_points(blocked,points,uncertainty_cells=1)
