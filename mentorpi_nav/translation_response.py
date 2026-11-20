"""Empirical signed translation response; command-to-settled-distance fit.

This does not identify friction, motor firmware, wheel contact or sensor scale.
"""
import math

DIRECTIONS=('forward','backward','left','right')


def validate_translation_response(profile):
    if profile.get('version') not in ('translation_deadzone_v1','translation_history_v1'):raise ValueError('Unknown translation response')
    if profile.get('version')=='translation_history_v1':
        if len(profile['breakaway_distance_m'])!=2 or any(not math.isfinite(v) or not 0<v<=.05 for v in profile['breakaway_distance_m']):raise ValueError('Invalid translation memory')
        if not math.isfinite(profile['idle_memory_decay_s']) or profile['idle_memory_decay_s']<=0:raise ValueError('Invalid memory decay')
    for name in DIRECTIONS:
        row=profile['directions'][name]
        if not math.isfinite(row['gain']) or not .1<=row['gain']<=3:raise ValueError('Invalid response gain')
        if not math.isfinite(row['deadband_mps']) or not 0<=row['deadband_mps']<.1:raise ValueError('Invalid response deadband')
    return profile


def response_rate(command,axis,profile):
    name=DIRECTIONS[2*axis+int(command<0)]
    row=profile['directions'][name]
    return math.copysign(row['gain']*max(0.,abs(command)-row['deadband_mps']),command)


def translation_response_torch(velocity,profile):
    import torch
    out=velocity.clone()
    for axis in range(2):
        positive,negative=(profile['directions'][name] for name in DIRECTIONS[2*axis:2*axis+2])
        gain=torch.where(velocity[:,axis]>=0,positive['gain'],negative['gain'])
        deadband=torch.where(velocity[:,axis]>=0,positive['deadband_mps'],negative['deadband_mps'])
        out[:,axis]=velocity[:,axis].sign()*gain*(velocity[:,axis].abs()-deadband).clamp_min(0)
    return out


class AxisTranslationResponse:
    """Scalar replay model. Signed memory is empirical, not literal wheel travel."""
    def __init__(self,profile,axis):self.profile=profile;self.axis=axis;self.memory=0.

    def step(self,command,dt):
        bound=self.profile['breakaway_distance_m'][self.axis]
        if abs(command)>1e-6:self.memory=max(-bound,min(bound,self.memory+command*dt))
        else:self.memory*=math.exp(-dt/self.profile['idle_memory_decay_s'])
        engaged=self.memory*command>0 and abs(self.memory)>=bound-1e-7
        return response_rate(command,self.axis,self.profile) if engaged else 0.


class BatchedTranslationResponse:
    def __init__(self,num_envs,device,profile):
        import torch
        self.profile=validate_translation_response(profile)
        self.memory=torch.zeros(num_envs,2,device=device)
        self.bounds=self.memory.new_tensor(profile['breakaway_distance_m'])

    def reset(self,ids):self.memory[ids]=0

    def step(self,velocity,dt):
        import torch
        command=velocity[:,:2]
        self.memory=torch.where(command.abs()>1e-6,(self.memory+command*dt).clamp(-self.bounds,self.bounds),
                                self.memory*math.exp(-dt/self.profile['idle_memory_decay_s']))
        engaged=(self.memory*command>0)&(self.memory.abs()>=self.bounds-1e-7)
        out=translation_response_torch(velocity,self.profile)
        out[:,:2]=torch.where(engaged,out[:,:2],0.)
        return out
