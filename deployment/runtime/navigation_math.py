"""ROS-independent map/pose conversion matching the training tensor contract."""
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


def occupancy_from_map(values,resolution,origin,base):
    """Nine-point conservative tile sampling. Unknown/outside/uncertain block.

    Source is ROS [height,width], origin/base are [x,y,yaw] in one frame.
    No footprint inflation is applied to the policy map.
    """
    values=np.asarray(values)
    axis=(49.5-np.arange(100))*.05
    forward,left=np.meshgrid(axis,axis,indexing='ij')
    blocked=np.zeros((100,100),dtype=bool)
    cb,sb=math.cos(base[2]),math.sin(base[2])
    co,so=math.cos(origin[2]),math.sin(origin[2])
    for dx in (-.0249,0.,.0249):
        for dy in (-.0249,0.,.0249):
            wx=base[0]+cb*(forward+dx)-sb*(left+dy)-origin[0]
            wy=base[1]+sb*(forward+dx)+cb*(left+dy)-origin[1]
            col=np.floor((co*wx+so*wy)/resolution).astype(int)
            row=np.floor((-so*wx+co*wy)/resolution).astype(int)
            inside=(row>=0)&(row<values.shape[0])&(col>=0)&(col<values.shape[1])
            samples=np.full((100,100),-1,dtype=np.int16)
            samples[inside]=values[row[inside],col[inside]]
            blocked|=(samples<0)|(samples>=50)
    return blocked


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


def subgoal_on_path(points,position,lookahead=1.):
    points=np.asarray(points)
    nearest=int(np.argmin(np.linalg.norm(points[:,:2]-np.asarray(position)[:2],axis=1)))
    distance=0.
    for i in range(nearest+1,len(points)):
        distance+=float(np.linalg.norm(points[i,:2]-points[i-1,:2]))
        if distance>=lookahead:return points[i,:2]
    return points[-1,:2]


def reachable_goal_candidates(values,resolution,origin,base,clearance=.25,min_distance=.5,max_distance=2.):
    import cv2
    known_free=((values>=0)&(values<50)).astype(np.uint8)
    safe=(cv2.distanceTransform(known_free,cv2.DIST_L2,5)*resolution>=clearance).astype(np.uint8)
    count,labels=cv2.connectedComponents(safe,connectivity=4)
    yy,xx=np.nonzero(safe)
    if len(xx)==0:return np.empty((0,2))
    c,s=math.cos(origin[2]),math.sin(origin[2])
    gx=(xx+.5)*resolution;gy=(yy+.5)*resolution
    points=np.column_stack((origin[0]+c*gx-s*gy,origin[1]+s*gx+c*gy))
    distance=np.linalg.norm(points-np.asarray(base[:2]),axis=1)
    closest=int(np.argmin(distance))
    component=labels[yy[closest],xx[closest]]
    mask=(labels[yy,xx]==component)&(distance>=min_distance)&(distance<=max_distance)
    return points[mask]


def map_alignment_score(values,resolution,origin,base,points):
    """Independent scan-to-saved-map consistency diagnostic, not a localizer."""
    import cv2
    points=np.asarray(points)
    keep=(np.linalg.norm(points[:,:2],axis=1)>.2)&(np.linalg.norm(points[:,:2],axis=1)<4.)
    points=points[keep]
    if len(points)<20:return {'points':len(points),'matched_fraction':0.,'median_error_m':float('inf')}
    c,s=math.cos(base[2]),math.sin(base[2])
    wx=base[0]+c*points[:,0]-s*points[:,1]-origin[0]
    wy=base[1]+s*points[:,0]+c*points[:,1]-origin[1]
    co,so=math.cos(origin[2]),math.sin(origin[2])
    col=np.floor((co*wx+so*wy)/resolution).astype(int)
    row=np.floor((-so*wx+co*wy)/resolution).astype(int)
    inside=(row>=0)&(row<values.shape[0])&(col>=0)&(col<values.shape[1])
    distance=cv2.distanceTransform((values<50).astype(np.uint8),cv2.DIST_L2,5)*resolution
    errors=np.full(len(points),float('inf'))
    errors[inside]=distance[row[inside],col[inside]]
    return {'points':len(points),'inside_fraction':float(inside.mean()),
            'matched_fraction':float((errors<=.15).mean()),'median_error_m':float(np.median(errors))}
