"""Ask Nav2 for a reachable path without any drive publishers."""
import json
import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from navigation_io import NavigationIO
from navigation_math import reachable_goal_candidates,yaw_of


def main():
    rclpy.init();node=NavigationIO('mentorpi_planner_probe')
    client=ActionClient(node,ComputePathToPose,'/compute_path_to_pose')
    deadline=time.monotonic()+12
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
    health=node.health()
    if health:raise RuntimeError(str(health))
    if not client.wait_for_server(timeout_sec=3):raise RuntimeError('Nav2 planner unavailable')
    base,_=node.pose()
    origin=node.map.info.origin;q=origin.orientation
    origin_pose=[origin.position.x,origin.position.y,yaw_of([q.x,q.y,q.z,q.w])]
    values=np.asarray(node.map.data,dtype=np.int16).reshape(node.map.info.height,node.map.info.width)
    candidates=reachable_goal_candidates(values,node.map.info.resolution,origin_pose,base)
    rng=np.random.default_rng(123)
    goal=candidates[rng.integers(len(candidates))]
    request=ComputePathToPose.Goal()
    request.goal.header.frame_id='map';request.goal.header.stamp=node.get_clock().now().to_msg()
    request.goal.pose.position.x=float(goal[0]);request.goal.pose.position.y=float(goal[1]);request.goal.pose.orientation.w=1.
    request.planner_id='GridBased';request.use_start=False
    future=client.send_goal_async(request)
    rclpy.spin_until_future_complete(node,future,timeout_sec=8)
    handle=future.result()
    if handle is None or not handle.accepted:raise RuntimeError('Planner rejected goal')
    future=handle.get_result_async()
    rclpy.spin_until_future_complete(node,future,timeout_sec=8)
    response=future.result()
    if response is None:raise RuntimeError('Planner timed out')
    points=[[p.pose.position.x,p.pose.position.y] for p in response.result.path.poses]
    result={'status':response.status,'path_points':len(points),'goal':goal.tolist(),'start':base.tolist(),'path':points,'drive_commands_published':0}
    from pathlib import Path
    Path('/home/ubuntu/shared/mentorpi_nav/diagnostics/planner_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='path'},indent=2),flush=True)
    node.destroy_node();rclpy.shutdown()
    if response.status!=4 or len(points)<2:raise RuntimeError('No successful usable path')


if __name__=='__main__':main()
