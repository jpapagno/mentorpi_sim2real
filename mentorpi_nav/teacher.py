"""Training-only grid planner used to diagnose feasibility and generate demonstrations.

It is never used by mentorpi_nav.cli.evaluate or the deployed/playback actor.
"""

import heapq
import math
import torch


def subgoal_on_path(points, position, lookahead=1.):
    """Select a lookahead point for simulation path-following comparisons."""
    import numpy as np
    points = np.asarray(points)
    nearest = int(np.argmin(np.linalg.norm(points[:, :2] - np.asarray(position)[:2], axis=1)))
    distance = 0.
    for index in range(nearest + 1, len(points)):
        distance += float(np.linalg.norm(points[index, :2] - points[index - 1, :2]))
        if distance >= lookahead:
            return points[index, :2]
    return points[-1, :2]

def line_clear(start, end, obstacles, radius, half_extent):
    if max(abs(v) for v in (*start, *end)) >= half_extent - radius:
        return False
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = max(dx*dx + dy*dy, 1e-12)
    for x, y, r in obstacles:
        t = max(0., min(1., ((x-start[0])*dx + (y-start[1])*dy) / length))
        if (x-start[0]-t*dx)**2 + (y-start[1]-t*dy)**2 <= (radius+r)**2:
            return False
    return True


def plan_path(start, goal, obstacles, cfg, margin=.08):
    resolution = .1
    limit = cfg.arena_half_extent - cfg.robot_radius - margin
    n = int(2*limit/resolution) + 1
    axis = [(i-(n-1)/2)*resolution for i in range(n)]
    inflation = cfg.robot_radius + margin + resolution/math.sqrt(2)
    free = {(i,j) for i in range(n) for j in range(n) if all(
        (axis[i]-x)**2+(axis[j]-y)**2 > (r+inflation)**2 for x,y,r in obstacles)}
    if not free:
        return None
    def nearest(point):
        return min(free, key=lambda p: (axis[p[0]]-point[0])**2+(axis[p[1]]-point[1])**2)
    begin, target = nearest(start), nearest(goal)
    queue = [(0., 0., begin)]
    costs, parent = {begin: 0.}, {}
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost > costs[node] + 1e-8:
            continue
        if node == target:
            nodes = [node]
            while nodes[-1] != begin:
                nodes.append(parent[nodes[-1]])
            points = [list(start)] + [[axis[i],axis[j]] for i,j in reversed(nodes)] + [list(goal)]
            # String-pull, using a continuous disc sweep rather than checking
            # only node occupancy. Preserve all corners when not shortcuttable.
            result, index = [points[0]], 0
            while index < len(points)-1:
                next_index = index+1
                for candidate in range(len(points)-1, index, -1):
                    if line_clear(points[index], points[candidate], obstacles, cfg.robot_radius+margin, cfg.arena_half_extent):
                        next_index = candidate
                        break
                if not line_clear(points[index], points[next_index], obstacles, cfg.robot_radius, cfg.arena_half_extent):
                    return None
                result.append(points[next_index])
                index = next_index
            return result
        i,j = node
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt = (i+di,j+dj)
            if nxt not in free or (di and dj and ((i+di,j) not in free or (i,j+dj) not in free)):
                continue
            next_cost = cost + math.hypot(di,dj)
            if next_cost < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = next_cost, node
                heapq.heappush(queue, (next_cost + math.hypot(nxt[0]-target[0],nxt[1]-target[1]), next_cost, nxt))
    # Tight reachable passages can require the original smaller clearance.
    return plan_path(start, goal, obstacles, cfg, margin=.02) if margin > .02 else None


class PathTeacher:
    def __init__(self, state):
        self.paths = []
        self.obstacles = []
        self.replan(state)

    def replan(self, state, active=None):
        positions, goals = state.xy.cpu().tolist(), state.goal.cpu().tolist()
        obstacles, valid = state.obstacles.cpu().tolist(), state.valid.cpu().tolist()
        self.obstacles = [[tuple(o) for o,v in zip(row,mask) if v] for row,mask in zip(obstacles,valid)]
        mask = [True]*len(positions) if active is None else active.cpu().tolist()
        old = self.paths if self.paths else [None]*len(positions)
        self.paths = [plan_path(p,g,o,state.cfg) if enabled else previous
                      for p,g,o,enabled,previous in zip(positions,goals,self.obstacles,mask,old)]

    def actions(self, state, active=None):
        positions, yaws = state.xy.cpu().tolist(), state.yaw.cpu().tolist()
        goals = state.goal.cpu().tolist()
        commands = []
        mask = [True]*len(positions) if active is None else active.cpu().tolist()
        for i, (position,yaw,goal,path) in enumerate(zip(positions,yaws,goals,self.paths)):
            if path is None or not mask[i]:
                commands.append([0.,0.,0.])
                continue
            # Always choose the farthest currently visible route waypoint.
            # This handles minor tracking errors without an opaque stateful
            # waypoint index that a feed-forward student cannot observe.
            target = None
            for point in reversed(path[1:]):
                if line_clear(position,point,self.obstacles[i],state.cfg.robot_radius+.08,state.cfg.arena_half_extent):
                    target = point
                    break
            if target is None:
                for point in reversed(path[1:]):
                    if line_clear(position,point,self.obstacles[i],state.cfg.robot_radius+.001,state.cfg.arena_half_extent):
                        target = point
                        break
            if target is None:
                path = self.paths[i] = plan_path(position,goal,self.obstacles[i],state.cfg)
                target = path[1] if path and len(path)>1 else position
            dx, dy = target[0]-position[0], target[1]-position[1]
            distance = math.hypot(dx,dy)
            speed = min(.18, .9*distance)
            vx = speed * (math.cos(yaw)*dx + math.sin(yaw)*dy) / max(distance,1e-9)
            vy = speed * (-math.sin(yaw)*dx + math.cos(yaw)*dy) / max(distance,1e-9)
            commands.append([vx/state.cfg.speed_limits[0],vy/state.cfg.speed_limits[1],0.])
        return torch.tensor(commands, device=state.device)
