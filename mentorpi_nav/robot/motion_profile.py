"""Measured body-response lookup and fixed feedback law; no hardware imports.

Profiles are experimental surface-specific fits, not motor firmware settings.
Interpolation is limited to the measured command range and actuator limits.
"""
import math
import statistics

DIRECTIONS=(('forward',0,1),('backward',0,-1),('left',1,1),('right',1,-1),('ccw',2,1),('cw',2,-1))
LIMITS=(.1,.1,.3)
VERSION='body_response_lookup_v1'


def calibration_path_clear(points,target_xy,margin=.02):
    """Conservative swept enclosing circle for short diagnostic targets only."""
    radius=math.hypot(.254,.2032)/2+margin
    dx,dy=target_xy;denom=dx*dx+dy*dy
    finite=[(x,y) for x,y in points if math.isfinite(x) and math.isfinite(y)]
    if len(finite)<20:return False
    for x,y in finite:
        t=max(0.,min(1.,(x*dx+y*dy)/denom)) if denom>0 else 0.
        if math.hypot(x-t*dx,y-t*dy)<=radius:return False
    return True


def validate_profile(profile):
    if profile.get('version')!=VERSION:raise ValueError('Unknown motion profile version')
    for name,axis,_ in DIRECTIONS:
        knots=profile['directions'][name]['knots']
        if not knots or knots[0]!=[0.,0.]:raise ValueError('Profile must start at zero')
        previous=(-1.,-1.)
        for command,response in knots:
            if not all(math.isfinite(v) for v in (command,response)) or not 0<=command<=LIMITS[axis] or response<0:
                raise ValueError('Invalid response knot')
            if command<=previous[0] or response<previous[1]:raise ValueError('Response knots must be monotonic')
            previous=command,response
    return profile


def identity_profile():
    return {'version':VERSION,'directions':{name:{'knots':[[0.,0.],[LIMITS[axis],LIMITS[axis]]],'minimum_moving_command':0.} for name,axis,_ in DIRECTIONS}}


def predicted_rate(knots,command):
    command=abs(command)
    for (x0,y0),(x1,y1) in zip(knots,knots[1:]):
        if command<=x1:return y0+(y1-y0)*(command-x0)/(x1-x0)
    return knots[-1][1]


def compensated_command(profile,desired):
    """Invert each signed response, capped at tested commands and hard limits.

    Below the first measured moving response, use the minimum observed moving
    command. Target tolerance in the fixed feedback law terminates this pulse;
    this does not claim arbitrary low-speed tracking is physically possible.
    """
    result=[]
    for axis,value in enumerate(desired):
        if not math.isfinite(value):raise ValueError('Nonfinite desired velocity')
        if abs(value)<1e-12:result.append(0.);continue
        name=DIRECTIONS[2*axis+(value<0)][0]
        direction=profile['directions'][name];knots=direction['knots']
        moving=[(u,v) for u,v in knots if v>(.005 if axis<2 else .01)]
        if not moving:raise ValueError('No measured moving response for '+name)
        command=moving[-1][0]
        for (u0,v0),(u1,v1) in zip(knots,knots[1:]):
            if v0<abs(value)<=v1:
                command=u0+(u1-u0)*(abs(value)-v0)/(v1-v0);break
        command=max(command,direction.get('minimum_moving_command',moving[0][0]))
        result.append(math.copysign(min(command,LIMITS[axis]),value))
    return result


def fixed_controller(body_error,yaw_error,profile):
    """Same proportional law for every profile: kp_xy=1,kp_yaw=1.5.

    Uses 1cm position and 2degree heading tolerances. Caller must validate the
    settled final pose; crossing a tolerance once is not acceptance.
    """
    if len(body_error)!=2 or not all(math.isfinite(v) for v in (*body_error,yaw_error)):
        raise ValueError('Invalid measured error')
    yaw_error=(yaw_error+math.pi)%(2*math.pi)-math.pi
    position_done=math.hypot(*body_error)<=.01
    heading_done=abs(yaw_error)<=math.radians(2)
    desired=[0. if position_done else max(-.08,min(.08,v)) for v in body_error]
    desired.append(0. if heading_done else max(-.15,min(.15,1.5*yaw_error)))
    return compensated_command(profile,desired),position_done and heading_done


def _slope(points):
    if len(points)<3:return None
    mt=statistics.mean(t for t,_ in points);mv=statistics.mean(v for _,v in points)
    denom=sum((t-mt)**2 for t,_ in points)
    return sum((t-mt)*(v-mv) for t,v in points)/denom if denom>1e-8 else None


def pulse_samples(record):
    """Fit trailing0.35s pose slopes, not command-integral ratios.

    Fused yaw and LiDAR displacement remain estimates, not external truth.
    Keep net pulse and cross-axis motion alongside these short-window rates.
    """
    from .calibrate_motion import body_delta
    runs=[];active=[]
    for row in record['trace']:
        if row['enabled']:active.append(row)
        elif active:runs.append(active);active=[]
    if active:runs.append(active)
    if len(runs)!=len(record['pulses']):raise ValueError('Pulse/trace mismatch')
    samples=[]
    for pulse,rows in zip(record['pulses'],runs):
        name=pulse['direction'];_,axis,sign=next(d for d in DIRECTIONS if d[0]==name)
        command=abs(pulse['requested_velocity'][axis]);origin=rows[0]['pose']
        tail=[r for r in rows if r['time']>=rows[-1]['time']-.35]
        slope=_slope([(r['time'],body_delta(origin,r['pose'])[axis]) for r in tail])
        if slope is None:raise ValueError('Insufficient measured samples')
        moving=abs(pulse['body_delta'][axis])>=(.01 if axis<2 else .02)
        samples.append({'direction':name,'command':command,'response':max(0.,sign*slope) if moving else 0.,
            'pulse_delta':pulse['body_delta'],'window_samples':len(tail),'moving':moving})
    return samples


def fit_profile(samples):
    profile={'version':VERSION,'status':'experimental_not_hardware_validated','directions':{},
             'measurement':'trailing0.35s fused pose slope; IMU-assisted yaw; surface-specific',
             'limitations':['short pulses','independent-axis fit','mixed motion unvalidated','stopping buzz unresolved']}
    for name,axis,_ in DIRECTIONS:
        groups={}
        for sample in samples:
            if sample['direction']==name:groups.setdefault(sample['command'],[]).append(sample['response'])
        if not groups:raise ValueError('Missing calibration direction: '+name)
        knots=[[0.,0.]];previous=0.
        for command,rates in sorted(groups.items()):
            previous=max(previous,statistics.median(rates))
            knots.append([command,previous])
        profile['directions'][name]={'knots':knots,'sample_count':sum(map(len,groups.values()))}
    return validate_profile(profile)
