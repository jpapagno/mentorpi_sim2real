"""Live map/TF/sensor/model integration; cannot publish a motor command."""
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
import torch
from navigation_io import NavigationIO
from navigation_math import reachable_goal_candidates,yaw_of


def main():
    torch.set_num_threads(1)
    root=Path('/home/ubuntu/shared/mentorpi_nav')
    rclpy.init();node=NavigationIO('mentorpi_policy_shadow_test')
    model=torch.jit.load(str(root/'models/policy.ts')).eval()
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.05)
    health=node.health()
    result={'health':health,'drive_commands_published':0}
    if not health:
        base,_=node.pose()
        origin=node.map.info.origin;q=origin.orientation
        origin_pose=[origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])]
        values=np.asarray(node.map.data,dtype=np.int16).reshape(node.map.info.height,node.map.info.width)
        candidates=reachable_goal_candidates(values,node.map.info.resolution,origin_pose,base)
        if len(candidates)==0:raise RuntimeError('No footprint-cleared reachable goal candidate')
        rng=np.random.default_rng(123)
        goal=candidates[rng.integers(len(candidates))]
        state,grid,details=node.observations(goal,[0.,0.,0.])
        with torch.no_grad():action=model(torch.from_numpy(state),torch.from_numpy(grid)).numpy()
        result.update(details,goal=goal.tolist(),candidate_goals=len(candidates),
                      state=state.tolist(),proposed_normalized_action=action.tolist())
        np.savez_compressed(root/'diagnostics/live_policy_case.npz',policy=state,occupancy=grid,action=action,
                            map=values,origin=origin_pose,resolution=node.map.info.resolution,base=base,goal=goal)
        preview=(1-grid[0,0])*255
        cv2.imwrite(str(root/'diagnostics/policy_grid.png'),preview.astype(np.uint8))
        display=np.full(values.shape,120,dtype=np.uint8)
        display[values==0]=255;display[values>=50]=0
        cv2.imwrite(str(root/'diagnostics/live_map.png'),np.flipud(display))
    (root/'diagnostics/shadow_test.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    node.destroy_node();rclpy.shutdown()
    return 0 if not health else 1


if __name__=='__main__':raise SystemExit(main())
