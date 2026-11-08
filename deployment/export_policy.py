"""Export the selected residual actor without Isaac Lab/RSL-RL dependencies."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict

from mentorpi_nav.core import NavigationCfg, NavigationState
from mentorpi_nav.playback import load_actor


class DeploymentActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.cnn=copy.deepcopy(actor.cnns['occupancy'])
        self.goal_encoder=copy.deepcopy(actor.goal_encoder)
        self.motion_encoder=copy.deepcopy(actor.motion_encoder)
        self.map_encoder=copy.deepcopy(actor.map_encoder)
        self.mlp=copy.deepcopy(actor.mlp)

    def forward(self,policy,occupancy):
        goal=policy[:,:2]
        norm=torch.linalg.vector_norm(goal,dim=-1,keepdim=True)
        unit=goal/norm.clamp_min(1e-6)
        goal_features=torch.cat((unit,norm),dim=-1)
        latent=torch.cat((self.goal_encoder(goal_features),self.motion_encoder(policy[:,2:8]),
                          self.map_encoder(self.cnn(occupancy))),dim=-1)
        speed=(.9*norm*(4*math.sqrt(2))/.2).clamp(max=.9)
        prior=torch.cat((unit*speed,torch.zeros_like(norm)),dim=-1)
        return (self.mlp(latent)+prior).clamp(-1,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='logs/mentorpi_navigation_best/final.pt')
    parser.add_argument('--output',default='outputs/robot_policy')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=Path(args.checkpoint).resolve()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(source.with_name('task.json').read_text())
    if manifest.get('network_architecture')!='goal_residual_cnn_v1':
        raise ValueError('Exporter supports goal_residual_cnn_v1 only')
    state=NavigationState(32,NavigationCfg(**manifest['navigation']),seed=89001)
    state.reset()
    obs=state.observations()
    actor=load_actor(source,obs)
    exported=DeploymentActor(actor).eval()
    with torch.no_grad():
        expected=actor(TensorDict(obs,batch_size=[32])).clamp(-1,1)
        actual=exported(obs['policy'],obs['occupancy'])
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        trace=torch.jit.trace(exported,(obs['policy'][:1],obs['occupancy'][:1]))
        torch.testing.assert_close(trace(obs['policy'],obs['occupancy']),expected,rtol=1e-5,atol=1e-6)
    trace.save(str(output/'policy.ts'))
    torch.onnx.export(exported,(obs['policy'][:1],obs['occupancy'][:1]),str(output/'policy.onnx'),
        input_names=['policy','occupancy'],output_names=['action'],opset_version=17,dynamo=False,
        dynamic_axes={'policy':{0:'batch'},'occupancy':{0:'batch'},'action':{0:'batch'}})
    import onnx
    onnx.checker.check_model(str(output/'policy.onnx'))
    np.savez_compressed(output/'parity_cases.npz',policy=obs['policy'].numpy(),
                        occupancy=obs['occupancy'].numpy(),action=expected.numpy())
    contract={'architecture':manifest['network_architecture'],'source_checkpoint':str(source),
        'policy_shape':[1,8],'occupancy_shape':[1,1,100,100],'action_shape':[1,3],
        'occupancy_resolution_m':.05,'goal_normalization_m':4*math.sqrt(2),
        'speed_limits':[.2,.2,.5],'occupancy_inflation_m':0.,
        'grid_top':'forward','grid_left':'left','unknown':'blocked',
        'torch_export_max_error':float((actual-expected).abs().max())}
    (output/'contract.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps(contract,indent=2))


if __name__=='__main__': main()
