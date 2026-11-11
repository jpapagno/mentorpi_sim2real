"""Render an actual saved policy rollout using the project's Isaac Kit scene.

Launch through Isaac Sim python.bat with the Isaac Lab PYTHONEXE override.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preview_bootstrap import configure_kit_usd
usd_root, dll_handles = configure_kit_usd()

import argparse
import asyncio
import json
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from mentorpi_nav.lab_env import MentorPiEnv, MentorPiEnvCfg
from mentorpi_nav.playback import NavigationPlayback

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='logs/mentorpi_navigation_best/final.pt')
parser.add_argument('--seed', type=int, default=789)
parser.add_argument('--output', default='outputs/readme_demo')
parser.add_argument('--limit', type=int, default=0)
add_launcher_args(parser)
args = parser.parse_args()
torch.set_num_threads(2)
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
session = NavigationPlayback(args.checkpoint, seed=args.seed)
cfg = MentorPiEnvCfg()
cfg.navigation = session.state.cfg
cfg.scene.num_envs = 1
cfg.sim.device = 'cpu'
cfg.asset_path = json.loads((ROOT / 'local_paths.json').read_text())['mentorpi_asset']

with launch_simulation(cfg, args):
    env = MentorPiEnv(cfg)
    env.reset()
    env.nav = session.state
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from PIL import Image
    viewport = get_active_viewport()
    from pxr import UsdGeom, Gf
    trail_curve = UsdGeom.BasisCurves.Define(env.sim.stage, '/World/PolicyTrail')
    trail_curve.CreateTypeAttr('linear')
    trail_curve.CreateWidthsAttr([0.018])
    trail_curve.SetWidthsInterpolation('constant')
    trail_curve.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.8, 0.9)])
    viewport.resolution = (960, 720)
    env.sim.set_camera_view(eye=(0.8, -1.6, 8.5), target=(0, 0, 0))
    env._sync_visuals()
    for _ in range(35):
        env.sim.render()
    records = []
    while True:
        env._sync_visuals()
        points = session.trail if len(session.trail) > 1 else session.trail * 2
        trail_curve.CreatePointsAttr([Gf.Vec3f(x,y,0.012) for x,y in points])
        trail_curve.CreateCurveVertexCountsAttr([len(points)])
        for _ in range(2):
            env.sim.render()
        path = out / f'scene_{len(records):04d}.png'
        capture = capture_viewport_to_file(viewport, str(path))
        future = asyncio.ensure_future(capture.wait_for_result())
        while not future.done():
            env.sim.render()
        future.result()
        grid = session.state.occupancy_grid()[0, 0].numpy()
        Image.fromarray(((1-grid)*255).astype('uint8')).save(out / f'grid_{len(records):04d}.png')
        records.append(dict(xy=session.state.xy[0].tolist(), yaw=float(session.state.yaw[0]),
                            goal=session.state.goal[0].tolist(), trail=list(session.trail),
                            time=session.episode_steps * cfg.navigation.dt,
                            distance=float(torch.linalg.vector_norm(session.state.goal-session.state.xy)),
                            status=session.status))
        print(f'CAPTURE {len(records)} {session.status}', flush=True)
        if session.terminal or (args.limit and len(records) >= args.limit):
            break
        for _ in range(2):
            session.advance()
    metadata = dict(checkpoint=args.checkpoint, seed=args.seed, status=session.status,
                    simulated_seconds=session.episode_steps*cfg.navigation.dt,
                    obstacle_count=int(session.state.valid.sum()), records=records)
    (out / 'rollout.json').write_text(json.dumps(metadata, indent=2))
    env.close()
