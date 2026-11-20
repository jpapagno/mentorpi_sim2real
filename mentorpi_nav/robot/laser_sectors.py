"""Versioned laser-sector observation contract, independent of ROS and Torch.

36 sectors start at -pi and proceed counterclockwise in the body frame.
Distances are body-origin endpoint distances; no-return sectors are zero
(unknown/blocked), rather than fictional clear space. This differs from the
legacy occupancy actor and must be paired with its own exported contract.
"""
import math
import numpy as np

SECTORS = 36
RANGE_SCALE = 2.5
GOAL_SCALE = 4 * math.sqrt(2)
OBS_SIZE = SECTORS + 2 + 3
CONTRACT = 'laser_sectors_36_goal2_action3_v1'


def sector_observation(points, body_goal, previous_action):
    points=np.asarray(points,dtype=np.float64).reshape(-1,2)
    distance=np.linalg.norm(points,axis=-1)
    good=np.isfinite(points).all(-1)&(distance>0)
    angle=np.arctan2(points[good,1],points[good,0])
    index=np.floor(((angle+math.pi)%(2*math.pi))*SECTORS/(2*math.pi)).astype(int).clip(0,SECTORS-1)
    sectors=np.full(SECTORS,np.inf)
    np.minimum.at(sectors,index,distance[good])
    sectors=np.where(np.isfinite(sectors),np.minimum(sectors,RANGE_SCALE)/RANGE_SCALE,0.)
    goal=np.asarray(body_goal,dtype=np.float64)
    result=np.concatenate((sectors,[np.linalg.norm(goal)/GOAL_SCALE,math.atan2(goal[1],goal[0])/math.pi],previous_action)).astype(np.float32)
    if result.shape!=(OBS_SIZE,) or not np.isfinite(result).all():raise ValueError('Invalid sector observation')
    return result


def sector_observation_torch(points, body_goal, previous_action):
    import torch
    distance=points.norm(dim=-1)
    good=torch.isfinite(points).all(-1)&(distance>0)
    safe=torch.where(good[...,None],points,0.)
    angle=torch.atan2(safe[...,1],safe[...,0])
    index=(((angle+math.pi)%(2*math.pi))*SECTORS/(2*math.pi)).floor().long().clamp(0,SECTORS-1)
    sectors=torch.full((len(points),SECTORS),float('inf'),device=points.device,dtype=points.dtype)
    sectors.scatter_reduce_(1,index,torch.where(good,distance,float('inf')),reduce='amin',include_self=True)
    sectors=torch.where(torch.isfinite(sectors),sectors.clamp(max=RANGE_SCALE)/RANGE_SCALE,0.)
    state=torch.stack((body_goal.norm(dim=-1)/GOAL_SCALE,torch.atan2(body_goal[:,1],body_goal[:,0])/math.pi),-1)
    return torch.cat((sectors,state,previous_action),-1).float()
