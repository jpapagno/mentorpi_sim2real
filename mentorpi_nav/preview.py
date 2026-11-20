"""Save the exact binary policy grid and a labeled preview without a GUI."""

from pathlib import Path

from PIL import Image, ImageDraw

from .core import world_to_body


def save_scene_preview(state, path, env_id=0):
    """Portable scene, first-return rays, and the exact binary actor input."""
    import math
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    grid=state.occupancy_grid()[env_id,0].cpu().numpy()
    image=Image.new('RGB',(1100,640),'#f8fafc');draw=ImageDraw.Draw(image)
    draw.text((25,15),'MentorPi 2D LiDAR navigation | shared shapes, rays and collision geometry',fill='#172033')
    draw.text((25,40),'World scene: boxes, doorway, concave outline and posts',fill='#172033')
    draw.text((590,40),'Policy input: white free / dark obstacle or unknown',fill='#172033')
    h=state.cfg.arena_half_extent;scale=500/(2*h)
    def pixel(p):return (25+(p[0]+h)*scale,70+(h-p[1])*scale)
    draw.rectangle((25,70,525,570),fill='#ffffff',outline='#64748b',width=3)
    vertices=state.polygons[env_id].cpu().tolist();counts=state.polygon_counts[env_id].cpu().tolist()
    for p,n in zip(vertices,counts):
        if n:draw.polygon([pixel(v) for v in p[:n]],fill='#dc7754',outline='#9a402a')
    for (x,y,r),valid in zip(state.obstacles[env_id].cpu().tolist(),state.valid[env_id].cpu().tolist()):
        if valid:
            cx,cy=pixel([x,y]);rr=r*scale;draw.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill='#dc7754')
    xy=state.xy[env_id].cpu().tolist();yaw=float(state.yaw[env_id]);c,s=math.cos(yaw),math.sin(yaw)
    if state.lidar:
        origin=state.cfg.lidar_origin
        sensor=[xy[0]+c*origin[0]-s*origin[1],xy[1]+s*origin[0]+c*origin[1]]
        for a,r in zip(state.lidar.angles.cpu().tolist()[::5],state.lidar.ranges[env_id].cpu().tolist()[::5]):
            if math.isfinite(r):
                end=[sensor[0]+r*math.cos(a+yaw),sensor[1]+r*math.sin(a+yaw)]
                if max(abs(v) for v in end)<=h+.03:draw.line([pixel(sensor),pixel(end)],fill='#b9d5e5')
    markers=[(state.goal[env_id].cpu().tolist(),state.cfg.goal_tolerance,'#22c55e')]
    if state.cfg.robot_size is not None:
        from .geometry import rectangle
        draw.polygon([pixel(p) for p in rectangle(xy,state.cfg.robot_size,yaw)],fill='#2387d9')
    else:
        markers.append((xy,state.cfg.robot_radius,'#2387d9'))
    for point,radius,color in markers:
        cx,cy=pixel(point);r=radius*scale;draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=color)
    draw.line([pixel(xy),pixel([xy[0]+.25*c,xy[1]+.25*s])],fill='#123a5a',width=3)
    raw=Image.fromarray(((1-grid)*255).astype('uint8')).convert('RGB').resize((500,500),Image.Resampling.NEAREST)
    image.paste(raw,(575,70));draw=ImageDraw.Draw(image)
    draw.ellipse((820,315,830,325),fill='#2387d9');draw.line((825,320,825,300),fill='#2387d9',width=3)
    draw.text((25,595),'2D analytic motion | no Isaac renderer or contact physics | colored markers are display overlays',fill='#475569')
    image.save(path)


def save_grid_preview(state, path, env_id=0):
    path = Path(path)
    data = state.occupancy_grid()[env_id, 0].detach().cpu().numpy()
    raw = Image.fromarray(((1 - data) * 255).astype("uint8"))
    raw.save(path.with_name(path.stem + "_binary.png"))
    size, margin, top = 600, 30, 65
    image = Image.new("RGB", (size + 2 * margin, size + top + 60), "#eeeeee")
    image.paste(raw.resize((size, size), Image.Resampling.NEAREST).convert("RGB"), (margin, top))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 12), "MentorPi policy occupancy: 100 x 100", fill="black")
    draw.text((margin, 32), f"0 free (white), 1 occupied (black); {state.cfg.occupancy_resolution:g} m/cell", fill="black")
    centre = (margin + size / 2, top + size / 2)
    draw.ellipse((centre[0]-5, centre[1]-5, centre[0]+5, centre[1]+5), fill="#1688ff")
    draw.line((centre[0], centre[1], centre[0], centre[1]-22), fill="#1688ff", width=3)
    goal = world_to_body((state.goal - state.xy)[env_id:env_id+1], state.yaw[env_id:env_id+1])[0].cpu().tolist()
    extent = state.cfg.occupancy_size * state.cfg.occupancy_resolution
    gx, gy = centre[0] - goal[1] / extent * size, centre[1] - goal[0] / extent * size
    if margin <= gx < margin + size and top <= gy < top + size:
        draw.ellipse((gx-6, gy-6, gx+6, gy+6), fill="#12be50")
    draw.text((margin, top + size + 12), "Top = forward; left = robot-left. Blue = robot; green = goal (when in view).", fill="black")
    draw.text((margin, top + size + 30), "Markers are preview overlays only; the policy grid contains binary cells only.", fill="black")
    image.save(path)
