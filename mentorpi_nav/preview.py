"""Save the exact binary policy grid and a labeled preview without a GUI."""

from pathlib import Path

from PIL import Image, ImageDraw

from .core import world_to_body


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
