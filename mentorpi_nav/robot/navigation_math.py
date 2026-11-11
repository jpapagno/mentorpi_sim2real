"""ROS-independent LiDAR and policy-state coordinate conversions."""
import math
import numpy as np

def yaw_of(q):
    x,y,z,w=q
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def rotation_of(q):
    x,y,z,w=q
    norm=math.sqrt(x*x+y*y+z*z+w*w)
    if norm<1e-8:raise ValueError('Invalid orientation quaternion')
    x,y,z,w=x/norm,y/norm,z/norm,w/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def overlay_points(blocked,points,uncertainty_cells=1):
    if len(points)==0:return blocked
    points=np.asarray(points)
    row=np.floor(50-points[:,0]/.05).astype(int)
    col=np.floor(50-points[:,1]/.05).astype(int)
    for dr in range(-uncertainty_cells,uncertainty_cells+1):
        for dc in range(-uncertainty_cells,uncertainty_cells+1):
            rr,cc=row+dr,col+dc
            valid=(rr>=0)&(rr<100)&(cc>=0)&(cc<100)
            blocked[rr[valid],cc[valid]]=True
    return blocked

def state_vector(goal,base,velocity,previous_action):
    dx,dy=goal[0]-base[0],goal[1]-base[1]
    c,s=math.cos(base[2]),math.sin(base[2])
    body_goal=np.array([c*dx+s*dy,-s*dx+c*dy])/(4*math.sqrt(2))
    result=np.concatenate((body_goal,np.asarray(velocity)/np.array([.2,.2,.5]),previous_action)).astype(np.float32)
    if result.shape!=(8,) or not np.isfinite(result).all():raise ValueError('Invalid policy state')
    return result
