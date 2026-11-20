"""Simple polygon geometry shared by sensing, disc sweeps and planning.

Polygons use world-space vertices in boundary order; convex and concave simple
polygons are supported. Circles retain their existing analytic representation.
"""
import math

import torch


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def edges(vertices, count):
    index = torch.arange(vertices.shape[-2], device=vertices.device)
    following = (index[None] + 1) % count[:, None].clamp_min(1)
    end = vertices.gather(1, following[..., None].expand(-1, -1, 2))
    return vertices, end, index[None] < count[:, None]


def point_segment_distance(p, a, b):
    delta = b-a
    t = ((p-a)*delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-15)
    return (p-a-t.clamp(0, 1)[..., None]*delta).norm(dim=-1)


def polygon_distance(points, vertices, counts):
    """Signed distance [B,N] to union: negative inside, +inf if empty."""
    result = torch.full(points.shape[:-1], float('inf'), device=points.device, dtype=points.dtype)
    for slot in range(vertices.shape[1]):
        a, b, valid = edges(vertices[:, slot], counts[:, slot])
        p = points[:, :, None]
        distance = point_segment_distance(p, a[:, None], b[:, None])
        distance = distance.masked_fill(~valid[:, None], float('inf')).amin(-1)
        ay, by, py = a[:, None, :, 1], b[:, None, :, 1], p[..., 1]
        denom = by-ay
        safe = torch.where(denom.abs() > 1e-15, denom, 1.)
        intercept = a[:, None, :, 0] + (py-ay)*(b-a)[:, None, :, 0]/safe
        crossings = ((ay > py) != (by > py)) & (p[..., 0] < intercept) & valid[:, None]
        inside = crossings.sum(-1) % 2 == 1
        result = torch.minimum(result, torch.where(inside, -distance, distance))
    return result


def polygon_raycast(origin, directions, vertices, counts):
    result = torch.full(directions.shape[:-1], float('inf'), device=directions.device, dtype=directions.dtype)
    for slot in range(vertices.shape[1]):
        a, b, valid = edges(vertices[:, slot].to(directions.dtype), counts[:, slot])
        delta, offset = (b-a)[:, None], a[:, None]-origin[:, None, None]
        denominator = cross(directions[:, :, None], delta)
        safe = torch.where(denominator.abs() > 1e-12, denominator, 1.)
        t = cross(offset, delta)/safe
        u = cross(offset, directions[:, :, None])/safe
        hit = valid[:, None] & (denominator.abs() > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
        result = torch.minimum(result, t.masked_fill(~hit, float('inf')).amin(-1))
    return result


def polygon_sweep(start, end, vertices, counts, radius):
    """Disc swept along a segment; detects tunneling, edges and interiors."""
    hit = (polygon_distance(start[:, None], vertices, counts)[:, 0] <= radius)
    hit |= polygon_distance(end[:, None], vertices, counts)[:, 0] <= radius
    motion = (end-start)[:, None]
    for slot in range(vertices.shape[1]):
        a, b, valid = edges(vertices[:, slot], counts[:, slot])
        edge = b-a
        offset = a-start[:, None]
        denominator = cross(motion, edge)
        safe = torch.where(denominator.abs() > 1e-12, denominator, 1.)
        t, u = cross(offset, edge)/safe, cross(offset, motion)/safe
        intersects = (denominator.abs() > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        distance = torch.minimum(point_segment_distance(start[:, None], a, b),
                                 point_segment_distance(end[:, None], a, b))
        distance = torch.minimum(distance, point_segment_distance(a, start[:, None], end[:, None]))
        distance = torch.minimum(distance, point_segment_distance(b, start[:, None], end[:, None]))
        hit |= ((intersects | (distance <= radius)) & valid).any(-1)
    return hit


def point_polygon_distance(point, polygon):
    """Scalar equivalent for CPU reset sampling and the training teacher."""
    x, y = point
    inside, distance = False, math.inf
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0., min(1., ((x-a[0])*dx+(y-a[1])*dy)/max(dx*dx+dy*dy, 1e-15)))
        distance = min(distance, math.hypot(x-a[0]-t*dx, y-a[1]-t*dy))
        if (a[1] > y) != (b[1] > y) and x < a[0]+(y-a[1])*dx/dy:
            inside = not inside
    return -distance if inside else distance


def segment_polygon_clear(start, end, polygon, radius):
    if min(point_polygon_distance(start, polygon), point_polygon_distance(end, polygon)) <= radius:
        return False
    # Distance between two segments: an intersection, or endpoint projection.
    def distance(p, a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(dx*dx+dy*dy, 1e-15)))
        return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        if (orient(start,end,a)*orient(start,end,b) <= 0 and orient(a,b,start)*orient(a,b,end) <= 0
            and max(min(start[0],end[0]),min(a[0],b[0])) <= min(max(start[0],end[0]),max(a[0],b[0]))
            and max(min(start[1],end[1]),min(a[1],b[1])) <= min(max(start[1],end[1]),max(a[1],b[1]))):
            return False
        if min(distance(start,a,b),distance(end,a,b),distance(a,start,end),distance(b,start,end)) <= radius:
            return False
    return True


def rectangle(center, size, yaw=0.):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[center[0]+c*x-s*y, center[1]+s*x+c*y]
            for x,y in [(-size[0]/2,-size[1]/2),(size[0]/2,-size[1]/2),
                        (size[0]/2,size[1]/2),(-size[0]/2,size[1]/2)]]


def rectangle_vertices(xy, yaw, size):
    local = xy.new_tensor([[-1,-1],[1,-1],[1,1],[-1,1]]) * xy.new_tensor(size)/2
    c,s = yaw.cos()[:,None],yaw.sin()[:,None]
    return xy[:,None] + torch.stack((c*local[:,0]-s*local[:,1],
                                    s*local[:,0]+c*local[:,1]),-1)


def rectangle_clearance(xy, yaw, size, obstacles, valid, polygons, counts, arena):
    """Exact positive separation; <=0 for overlap/touch, including concave edges."""
    corners = rectangle_vertices(xy,yaw,size)
    result = (arena-corners.abs()).amin(dim=(-1,-2))
    c,s=yaw.cos()[:,None],yaw.sin()[:,None]
    offset=obstacles[...,:2]-xy[:,None]
    local=torch.stack((c*offset[...,0]+s*offset[...,1],-s*offset[...,0]+c*offset[...,1]),-1)
    q=local.abs()-xy.new_tensor(size)/2
    circle_distance=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)-obstacles[...,2]
    result=torch.minimum(result,circle_distance.masked_fill(~valid,float('inf')).amin(-1))
    rect_counts=torch.full((len(xy),1),4,device=xy.device,dtype=torch.long)
    result=torch.minimum(result,polygon_distance(corners,polygons,counts).amin(-1))
    for slot in range(polygons.shape[1]):
        a,b,edge_valid=edges(polygons[:,slot],counts[:,slot])
        vertex_distance=polygon_distance(a,corners[:,None],rect_counts)
        result=torch.minimum(result,vertex_distance.masked_fill(~edge_valid,float('inf')).amin(-1))
        for edge in range(4):
            start,end=corners[:,edge,None],corners[:,(edge+1)%4,None]
            distance=torch.minimum(point_segment_distance(start,a,b),point_segment_distance(end,a,b))
            distance=torch.minimum(distance,point_segment_distance(a,start,end))
            distance=torch.minimum(distance,point_segment_distance(b,start,end))
            motion=end-start;delta=b-a;den=cross(motion,delta)
            safe=torch.where(den.abs()>1e-12,den,1.)
            t,u=cross(a-start,delta)/safe,cross(a-start,motion)/safe
            intersects=(den.abs()>1e-12)&(t>=0)&(t<=1)&(u>=0)&(u<=1)
            distance=torch.where(intersects,0.,distance)
            result=torch.minimum(result,distance.masked_fill(~edge_valid,float('inf')).amin(-1))
    return result


def rectangle_sweep(start, end, yaw, end_yaw, size, obstacles, valid, polygons, counts, arena):
    """Conservative continuous translation/rotation sweep, not endpoint-only.

    Nine poses cover each step. A Lipschitz motion bound pads their clearance
    by half the inter-pose travel of the furthest corner, preventing tunneling.
    """
    samples=9
    t=torch.linspace(0,1,samples,device=start.device,dtype=start.dtype)
    xy=start[:,None]+(end-start)[:,None]*t[None,:,None]
    angles=yaw[:,None]+(end_yaw-yaw)[:,None]*t
    def repeat(value):return value.repeat_interleave(samples,dim=0)
    clearance=rectangle_clearance(xy.reshape(-1,2),angles.flatten(),size,repeat(obstacles),
        repeat(valid),repeat(polygons),repeat(counts),arena).reshape(-1,samples)
    radius=math.hypot(*size)/2
    padding=((end-start).norm(dim=-1)+radius*(end_yaw-yaw).abs())/(2*(samples-1))
    return (clearance.amin(-1)<=padding+1e-7)
