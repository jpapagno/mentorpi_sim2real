"""Local scan/map registration for stationary recovery after manual movement."""
import math
import numpy as np
from navigation_math import map_alignment_score

def fit_scan_pose(values,resolution,origin,points,prior):
    import cv2
    points=np.asarray(points)[:,:2]
    keep=(np.linalg.norm(points,axis=1)>.2)&(np.linalg.norm(points,axis=1)<4.)
    points=points[keep]
    if len(points)<100:raise ValueError('At least 100 scan points required')
    values=np.asarray(values)
    field=cv2.distanceTransform((values<50).astype(np.uint8),cv2.DIST_L2,5)*resolution
    co,so=math.cos(origin[2]),math.sin(origin[2])
    prior=np.asarray(prior,dtype=float)
    center=prior.copy();best_cost=float('inf')
    for xy_span,xy_step,yaw_span,yaw_step in ((.3,.05,.6,.05),(.05,.01,.05,.01)):
        xs=center[0]+np.arange(-xy_span,xy_span+xy_step*.5,xy_step)
        ys=center[1]+np.arange(-xy_span,xy_span+xy_step*.5,xy_step)
        xx,yy=np.meshgrid(xs,ys);xx=xx.ravel();yy=yy.ravel()
        best=center.copy();best_cost=float('inf')
        for yaw in center[2]+np.arange(-yaw_span,yaw_span+yaw_step*.5,yaw_step):
            c,s=math.cos(yaw),math.sin(yaw)
            wx=xx[:,None]+c*points[:,0]-s*points[:,1]-origin[0]
            wy=yy[:,None]+s*points[:,0]+c*points[:,1]-origin[1]
            col=np.floor((co*wx+so*wy)/resolution).astype(int)
            row=np.floor((-so*wx+co*wy)/resolution).astype(int)
            inside=(row>=0)&(row<values.shape[0])&(col>=0)&(col<values.shape[1])
            errors=np.full(row.shape,.6,dtype=float)
            errors[inside]=np.minimum(.6,field[row[inside],col[inside]])
            count=max(1,int(len(points)*.8))
            robust=np.partition(errors,count-1,axis=1)[:,:count].mean(axis=1)
            cost=robust+.005*np.hypot(xx-prior[0],yy-prior[1])+.001*abs(yaw-prior[2])
            index=int(np.argmin(cost))
            if cost[index]<best_cost:
                best_cost=float(cost[index]);best=np.array([xx[index],yy[index],yaw])
        center=best
    return {'pose':center.tolist(),'robust_cost':best_cost,
            'before':map_alignment_score(values,resolution,origin,prior,points),
            'after':map_alignment_score(values,resolution,origin,center,points)}
