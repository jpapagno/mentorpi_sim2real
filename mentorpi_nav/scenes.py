"""Portable 2D scene definitions; no renderer or simulator dependencies."""
import math

from .geometry import rectangle, segment_polygon_clear


class SceneSamplingError(RuntimeError):
    """A valid procedural scene could not be found within this batch of attempts."""


def validate_polygon(vertices, capacity):
    if not 3 <= len(vertices) <= capacity or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in vertices):
        raise ValueError('Polygon needs 3..capacity finite xy vertices')
    area = sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(vertices,vertices[1:]+vertices[:1]))
    if abs(area) < 1e-9:
        raise ValueError('Degenerate polygon')
    for i,a in enumerate(vertices):
        b = vertices[(i+1)%len(vertices)]
        if math.dist(a,b) < 1e-8:
            raise ValueError('Duplicate polygon vertices')
        for j in range(i+1,len(vertices)):
            if j == i+1 or (i == 0 and j == len(vertices)-1):
                continue
            # A two-vertex closed boundary is sufficient for the segment test.
            if not segment_polygon_clear(a,b,[vertices[j],vertices[(j+1)%len(vertices)]],0.):
                raise ValueError('Self-intersecting polygon')


def sample_scene(cfg, rng):
    spec = cfg.scene_spec
    layout = spec.get('layout', 'fixed')
    if layout=='mixture':
        from dataclasses import replace
        choices=spec['choices']
        if not choices or any(choice.get('weight',1)<=0 for choice in choices):raise ValueError('Scene mixture needs positive weights')
        choice=rng.choices(choices,weights=[choice.get('weight',1) for choice in choices],k=1)[0]
        for _ in range(10):
            try:return sample_scene(replace(cfg,scene_spec=choice['scene']),rng)
            except SceneSamplingError:pass
        raise SceneSamplingError(f"Could not sample mixture scene {choice.get('name','unnamed')} after 1000 attempts")
    from .teacher import plan_path, line_clear
    for _ in range(100):
        circles, polygons = [], []
        if layout == 'fixed':
            start, goal, yaw = spec['start'], spec['goal'], spec.get('yaw',0.)
            for shape in spec.get('obstacles', []):
                if shape['type'] == 'circle':
                    circles.append([*shape['center'],shape['radius']])
                elif shape['type'] == 'rectangle':
                    if len(shape['size']) != 2 or min(shape['size']) <= 0:
                        raise ValueError('Rectangle sizes must be positive')
                    polygons.append(rectangle(shape['center'],shape['size'],shape.get('yaw',0.)))
                elif shape['type'] == 'polygon':
                    polygons.append(shape['vertices'])
                else:
                    raise ValueError('Unknown scene shape')
        elif layout in ('rooms','clutter'):
            h = cfg.arena_half_extent
            gap = rng.uniform(*spec.get('door_width_range',[.5,.7]))
            door_y = rng.uniform(*spec.get('door_center_range',[-.4,.4]))
            wall_x = rng.uniform(*spec.get('wall_x_range',[-.2,.2]))
            thickness = spec.get('wall_thickness',.12)
            if layout=='rooms':
                for lo,hi in [(-h,door_y-gap/2),(door_y+gap/2,h)]:
                    polygons.append(rectangle([wall_x,(lo+hi)/2],[thickness,hi-lo]))
            start = [-h+.5,rng.uniform(-h+.5,h-.5)]
            goal = [h-.5,rng.uniform(-h+.5,h-.5)]
            if layout=='clutter':
                start=[rng.uniform(-h+.25,h-.25) for _ in range(2)]
                goal=[rng.uniform(-h+.25,h-.25) for _ in range(2)]
            yaw = rng.uniform(-math.pi,math.pi)
            for _ in range(rng.randint(*spec['rectangle_count_range'])):
                size = [rng.uniform(*spec['rectangle_size_range']) for _ in range(2)]
                polygons.append(rectangle([rng.uniform(-h+.5,h-.5),rng.uniform(-h+.5,h-.5)],size,rng.uniform(-math.pi,math.pi)))
            for _ in range(rng.randint(*spec.get('concave_count_range',[0,0]))):
                size = rng.uniform(*spec['rectangle_size_range'])
                local = [[0,0],[size,0],[size,size*.3],[size*.3,size*.3],[size*.3,size],[0,size]]
                cx,cy=rng.uniform(-h+.6,h-.6),rng.uniform(-h+.6,h-.6)
                angle=rng.uniform(-math.pi,math.pi);c,s=math.cos(angle),math.sin(angle)
                polygons.append([[cx+c*x-s*y,cy+s*x+c*y] for x,y in local])
            for _ in range(rng.randint(*spec.get('post_count_range',[0,0]))):
                circles.append([rng.uniform(-h+.3,h-.3),rng.uniform(-h+.3,h-.3),rng.uniform(*spec.get('post_radius_range',[.025,.06]))])
        else:
            raise ValueError('Unknown scene layout')
        if len(circles) > cfg.max_obstacles or len(polygons) > cfg.max_polygons:
            raise ValueError('Scene exceeds geometry capacity')
        for polygon in polygons:
            validate_polygon(polygon,cfg.max_polygon_vertices)
        if any(len(c) != 3 or not all(math.isfinite(v) for v in c) or c[2] <= 0 for c in circles):
            raise ValueError('Invalid circle')
        if len(start) != 2 or len(goal) != 2 or not all(math.isfinite(v) for v in [*start,*goal,yaw]):
            raise ValueError('Invalid start/goal/heading')
        safe = all(line_clear(p,p,circles,cfg.robot_radius+cfg.planning_margin,cfg.arena_half_extent,polygons)
                   for p in [start,goal])
        if safe and math.dist(start,goal) >= cfg.min_goal_distance and plan_path(start,goal,circles,cfg,polygons=polygons) is not None:
            return start,goal,circles,yaw,polygons
        if layout == 'fixed':
            raise ValueError('Fixed scene has an unsafe start/goal or no reachable route')
    raise SceneSamplingError('Could not sample a reachable scene in 100 attempts; reduce clutter')
