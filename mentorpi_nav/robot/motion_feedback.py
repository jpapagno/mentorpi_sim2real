"""Measured planar velocity and commanded-without-progress detection."""
from collections import deque
import math

def angle_delta(new,old):return (new-old+math.pi)%(2*math.pi)-math.pi

class PlanarVelocityEstimator:
    def __init__(self):self.samples=deque(maxlen=7)
    def add(self,stamp,x,y,yaw):
        if not all(math.isfinite(v) for v in (stamp,x,y,yaw)):
            self.samples.clear();return None
        if self.samples:
            last=self.samples[-1];dt=stamp-last[0]
            if dt<=0:return None
            if dt>.5 or math.hypot(x-last[1],y-last[2])>max(.05,dt):self.samples.clear()
        unwrapped=yaw if not self.samples else self.samples[-1][3]+angle_delta(yaw,self.samples[-1][3])
        self.samples.append((stamp,x,y,unwrapped))
        while len(self.samples)>3 and stamp-self.samples[0][0]>.4:self.samples.popleft()
        if len(self.samples)<3 or stamp-self.samples[0][0]<.1:return None
        means=[sum(row[i] for row in self.samples)/len(self.samples) for i in range(4)]
        denominator=sum((row[0]-means[0])**2 for row in self.samples)
        slopes=[sum((row[0]-means[0])*(row[i]-means[i]) for row in self.samples)/denominator for i in (1,2,3)]
        c,s=math.cos(yaw),math.sin(yaw)
        velocity=(c*slopes[0]+s*slopes[1],-s*slopes[0]+c*slopes[1],slopes[2])
        if math.hypot(*velocity[:2])>1 or abs(velocity[2])>3:
            self.samples.clear();return None
        return velocity

class MotionProgressMonitor:
    def __init__(self,seconds=2.5,minimum_progress=.015):
        self.seconds=seconds;self.minimum_progress=minimum_progress;self.start=None
    def update(self,now,pose,applied):
        if not all(math.isfinite(v) for v in (now,*pose,*applied)):return 'invalid_motion_feedback'
        effort=math.hypot(*applied[:2])+.14*abs(applied[2])
        if effort<.02:self.start=None;return None
        if self.start is None:self.start=(now,*pose);return None
        progress=math.hypot(pose[0]-self.start[1],pose[1]-self.start[2])+.14*abs(angle_delta(pose[2],self.start[3]))
        if progress>=self.minimum_progress:self.start=(now,*pose);return None
        if now-self.start[0]>=self.seconds:return 'commanded_without_measured_motion'
        return None
