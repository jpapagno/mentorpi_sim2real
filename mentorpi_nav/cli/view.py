"""Lightweight Tkinter navigation viewer; does not import Isaac Sim or Isaac Lab."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import re
import pickle

import torch

from mentorpi_nav.core import world_to_body
from mentorpi_nav.playback import NavigationPlayback


def checkpoint_choices(directory):
    """Numeric iteration order, followed by final.pt when present."""
    directory = Path(directory)
    models = [p for p in directory.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)]
    models.sort(key=lambda p: int(p.stem.split("_")[-1]))
    final = directory / "final.pt"
    return models + ([final] if final.is_file() else [])


class NavigationViewer:
    def __init__(self, session, validate_steps=0, report=None, seed=123):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.session, self.validate_steps, self.report = session, validate_steps, report
        self.seed = seed
        self.root = tk.Tk()
        self.root.title("MentorPi — lightweight policy viewer")
        self.root.configure(bg="#18222d")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        self.keys = set()
        self.frames = 0
        self.resets = 0
        self.started = time.perf_counter()
        self.map_photo = None
        self.paused = tk.BooleanVar(value=False)
        self.auto_reset = tk.BooleanVar(value=False)
        self.speed = tk.DoubleVar(value=1.0)
        self.mode = tk.StringVar(value=session.mode)
        self.info = tk.StringVar()
        self.command = tk.StringVar()
        self.watch = tk.BooleanVar(value=False)
        self.selected_checkpoint = tk.StringVar(value=session.checkpoint.name if session.checkpoint else "")
        self._watched_candidate = None
        self._loaded_fingerprint = self.checkpoint_fingerprint(session.checkpoint)

        title = "MentorPi navigation | " + (session.checkpoint.name if session.checkpoint else "manual playground")
        ttk.Label(self.root, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        ttk.Label(self.root, text="Arena view and exact robot-relative policy input — no Isaac renderer").pack(anchor="w", padx=16)
        self.scenario_note = tk.StringVar(value=session.scenario_description)
        ttk.Label(self.root, textvariable=self.scenario_note).pack(anchor="w", padx=16, pady=(3, 0))
        checkpoints = ttk.Frame(self.root)
        checkpoints.pack(fill="x", padx=16, pady=(8, 0))
        ttk.Label(checkpoints, text="Checkpoint").pack(side="left", padx=(0, 8))
        self.checkpoint_selector = ttk.Combobox(checkpoints, textvariable=self.selected_checkpoint, state="readonly", width=24)
        self.checkpoint_selector.pack(side="left")
        self.checkpoint_selector.bind("<<ComboboxSelected>>", lambda event: self.select_checkpoint())
        ttk.Button(checkpoints, text="Refresh", command=self.refresh_checkpoints).pack(side="left", padx=6)
        ttk.Checkbutton(checkpoints, text="Follow newest checkpoint", variable=self.watch).pack(side="left", padx=8)
        self.checkpoint_note = tk.StringVar(value=str(session.checkpoint.parent) if session.checkpoint else "No checkpoint loaded")
        ttk.Label(checkpoints, textvariable=self.checkpoint_note, wraplength=500).pack(side="left", padx=4)
        self.refresh_checkpoints()
        panels = ttk.Frame(self.root)
        panels.pack(padx=16, pady=12)
        self.world = tk.Canvas(panels, width=560, height=560, bg="#202e3b", highlightthickness=0)
        self.world.grid(row=0, column=0, padx=(0, 14))
        self.grid = tk.Canvas(panels, width=440, height=560, bg="#202e3b", highlightthickness=0)
        self.grid.grid(row=0, column=1)
        self.world.bind("<Button-1>", self.click_goal)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=16)
        ttk.Checkbutton(controls, text="Pause", variable=self.paused).pack(side="left", padx=(0, 12))
        ttk.Button(controls, text="Step", command=self.step_once).pack(side="left", padx=4)
        ttk.Button(controls, text="New episode", command=self.new_episode).pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="Auto-reset", variable=self.auto_reset).pack(side="left", padx=12)
        choices = ["policy", "manual", "random"] if session.actor is not None else ["manual", "random"]
        selector = ttk.Combobox(controls, textvariable=self.mode, values=choices, state="readonly", width=8)
        selector.pack(side="left", padx=8)
        selector.bind("<<ComboboxSelected>>", lambda event: self.change_mode())
        ttk.Label(controls, text="Speed").pack(side="left", padx=(12, 2))
        ttk.Scale(controls, variable=self.speed, from_=0.25, to=4.0, length=150).pack(side="left")
        ttk.Label(self.root, textvariable=self.info).pack(anchor="w", padx=16, pady=(10, 2))
        ttk.Label(self.root, textvariable=self.command).pack(anchor="w", padx=16)
        ttk.Label(self.root, text="Manual: W/S forward/back • A/D strafe • Q/E turn | Click clear arena space to move goal").pack(anchor="w", padx=16, pady=(4, 12))
        self.root.bind("<KeyPress>", lambda event: self.keys.add(event.keysym.lower()))
        self.root.bind("<KeyRelease>", lambda event: self.keys.discard(event.keysym.lower()))
        self.root.bind("<FocusOut>", lambda event: self.keys.clear())
        self.root.after(0, self.tick)
        self.root.after(1000, self.poll_checkpoints)

    @staticmethod
    def checkpoint_fingerprint(path):
        if path is None:
            return None
        path = Path(path)
        stat = path.stat()
        return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    def refresh_checkpoints(self):
        paths = checkpoint_choices(self.session.checkpoint.parent) if self.session.checkpoint else []
        self.checkpoint_selector.configure(values=[p.name for p in paths])
        return paths

    def load_checkpoint(self, path):
        # Construct first so a partly written/invalid checkpoint cannot replace
        # the running session. Reuse the seed for fair visual comparisons.
        new_session = NavigationPlayback(path, seed=self.seed, obstacle_count=self.session.obstacle_override)
        self.session = new_session
        self.mode.set("policy")
        self.scenario_note.set(new_session.scenario_description)
        self.selected_checkpoint.set(new_session.checkpoint.name)
        self._loaded_fingerprint = self.checkpoint_fingerprint(new_session.checkpoint)
        self.checkpoint_note.set(f"Loaded {new_session.checkpoint.name}; reset to seed {self.seed}")
        self.root.title(f"MentorPi policy viewer — {new_session.checkpoint.name}")
        self.keys.clear()
        self.draw()

    def select_checkpoint(self):
        if not self.session.checkpoint:
            return
        try:
            self.load_checkpoint(self.session.checkpoint.parent / self.selected_checkpoint.get())
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, pickle.UnpicklingError) as error:
            from tkinter import messagebox
            messagebox.showerror("Cannot load checkpoint", str(error), parent=self.root)

    def poll_checkpoints(self):
        if not self.running:
            return
        try:
            paths = self.refresh_checkpoints()
            if self.watch.get() and paths:
                latest = max(paths, key=lambda p: p.stat().st_mtime_ns)
                fingerprint = self.checkpoint_fingerprint(latest)
                # Require size/mtime to remain stable across two polls. If a
                # save is still incomplete, keep the current model and retry.
                if fingerprint == self._watched_candidate and fingerprint != self._loaded_fingerprint:
                    self.load_checkpoint(latest)
                self._watched_candidate = fingerprint
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, pickle.UnpicklingError):
            self.checkpoint_note.set("Waiting for checkpoint write to finish...")
        self.root.after(1000, self.poll_checkpoints)

    def change_mode(self):
        self.session.mode = self.mode.get()
        self.keys.clear()

    def manual_action(self):
        keys = self.keys
        return [float("w" in keys or "up" in keys) - float("s" in keys or "down" in keys),
                float("a" in keys or "left" in keys) - float("d" in keys or "right" in keys),
                float("q" in keys) - float("e" in keys)]

    def new_episode(self):
        self.session.reset()
        self.resets += 1
        self.keys.clear()
        self.draw()

    def step_once(self):
        self.paused.set(True)
        if not self.session.terminal:
            self.session.advance(self.manual_action())
        self.draw()

    def click_goal(self, event):
        scale = 480 / (2 * self.session.state.cfg.arena_half_extent)
        point = ((event.x - 280) / scale, (280 - event.y) / scale)
        self.session.move_goal(point)
        self.draw()

    def tick(self):
        if not self.running:
            return
        start = time.perf_counter()
        if not self.paused.get():
            if self.session.terminal and self.auto_reset.get():
                self.session.reset()
            if not self.session.terminal:
                self.session.advance(self.manual_action())
        self.draw()
        self.frames += 1
        if self.validate_steps and self.frames == max(1, self.validate_steps // 2):
            self.new_episode()
        if self.validate_steps and self.frames >= self.validate_steps:
            self.close()
            return
        delay = self.session.state.cfg.dt / self.speed.get() - (time.perf_counter() - start)
        self.root.after(max(1, int(delay * 1000)), self.tick)

    def draw(self):
        state = self.session.state
        h = state.cfg.arena_half_extent
        scale = 480 / (2*h)

        def pixel(x, y):
            return 280 + x*scale, 280 - y*scale

        def circle(canvas, xy, radius, **kwargs):
            x, y = pixel(*xy)
            r = radius * scale
            canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

        canvas = self.world
        canvas.delete("all")
        canvas.create_text(18, 18, text="ARENA · world frame", fill="#d9e7ef", anchor="w")
        canvas.create_rectangle(40, 40, 520, 520, fill="#edf2f5", outline="#8396a6", width=3)
        for i in range(1, 10):
            p = 40 + i*48
            canvas.create_line(p, 40, p, 520, fill="#dfe6eb")
            canvas.create_line(40, p, 520, p, fill="#dfe6eb")
        for obstacle, valid in zip(state.obstacles[0].cpu().tolist(), state.valid[0].cpu().tolist()):
            if valid:
                circle(canvas, obstacle[:2], obstacle[2], fill="#dc7754", outline="#b85335", width=2)
        if len(self.session.trail) > 1:
            points = [v for xy in self.session.trail for v in pixel(*xy)]
            canvas.create_line(*points, fill="#7ba6c2", width=2)
        goal = state.goal[0].cpu().tolist()
        circle(canvas, goal, state.cfg.goal_tolerance, fill="#a6e6bc", outline="#22a65d", width=2)
        gx, gy = pixel(*goal)
        canvas.create_text(gx, gy, text="G", fill="#135f35", font=("Segoe UI", 10, "bold"))
        xy = state.xy[0].cpu().tolist()
        circle(canvas, xy, state.cfg.robot_radius, fill="#2088ce", outline="#125e94", width=2)
        yaw = float(state.yaw[0])
        rx, ry = pixel(*xy)
        end = pixel(xy[0] + 0.25*math.cos(yaw), xy[1] + 0.25*math.sin(yaw))
        canvas.create_line(rx, ry, *end, fill="#103c60", width=3, arrow="last")
        canvas.create_text(18, 542, text="Blue: robot / heading   Green: goal   Orange: obstacle", fill="#d9e7ef", anchor="w")

        occupancy = state.occupancy_grid()[0, 0].cpu().numpy()
        image = self.Image.fromarray(((1-occupancy)*255).astype("uint8")).resize((400, 400), self.Image.Resampling.NEAREST)
        self.map_photo = self.ImageTk.PhotoImage(image)
        canvas = self.grid
        canvas.delete("all")
        canvas.create_text(20, 18, text="POLICY INPUT · 100 × 100", fill="#d9e7ef", anchor="w")
        canvas.create_image(20, 70, image=self.map_photo, anchor="nw")
        canvas.create_oval(217, 267, 223, 273, fill="#2088ce", outline="")
        canvas.create_line(220, 270, 220, 253, fill="#2088ce", width=2, arrow="last")
        relative_goal = world_to_body(state.goal - state.xy, state.yaw)[0].cpu().tolist()
        extent = state.cfg.occupancy_resolution * 100
        gx, gy = 220-relative_goal[1]/extent*400, 270-relative_goal[0]/extent*400
        if 20 <= gx < 420 and 70 <= gy < 470:
            canvas.create_oval(gx-4, gy-4, gx+4, gy+4, fill="#22c565", outline="")
        canvas.create_text(20, 495, text=f"White = clear, black = occupied | {state.cfg.occupancy_resolution:g} m / tile", fill="#d9e7ef", anchor="w")
        canvas.create_text(20, 519, text="Top = robot forward. Colored markers are display overlays.", fill="#d9e7ef", anchor="w")
        distance = float(torch.linalg.vector_norm(state.goal - state.xy))
        self.info.set(f"{self.session.status}  |  episode {self.session.episode_steps*state.cfg.dt:.1f} / {state.cfg.episode_seconds:g} s  |  goal {distance:.2f} m  |  return {self.session.episode_return:.2f}")
        velocity = state.velocity[0].cpu().tolist()
        action = state.actions[0].cpu().tolist()
        self.command.set(f"Command [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}]   Actual vx {velocity[0]:+.3f} m/s · vy {velocity[1]:+.3f} m/s · yaw {velocity[2]:+.3f} rad/s")

    def close(self):
        if not self.running:
            return
        self.running = False
        result = {
            "frames": self.frames, "steps": self.session.total_steps, "resets": self.resets, "mode": self.session.mode,
            "status": self.session.status, "checkpoint": str(self.session.checkpoint),
            "elapsed_s": time.perf_counter() - self.started,
            "evaluation_obstacles": self.session.state.cfg.obstacle_count,
            "training_obstacles": self.session.training_cfg.obstacle_count,
            "obstacle_override": self.session.obstacle_override,
            "simulator_modules_loaded": any(name.split(".")[0] in {"isaaclab", "isaacsim", "pxr", "newton", "omni"} for name in sys.modules),
        }
        if self.report:
            path = Path(self.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2] / "logs/grid_empty_room/final.pt"
    parser.add_argument("--checkpoint", default=str(default) if default.exists() else None)
    parser.add_argument("--run-dir", default=None, help="Open the newest checkpoint from a training log directory")
    parser.add_argument("--follow", action="store_true", help="Automatically load new checkpoints from the same run")
    parser.add_argument("--obstacles", type=int, default=None,
                        help="Override obstacle count for viewer testing (0 to checkpoint max_obstacles)")
    parser.add_argument("--manual", action="store_true", help="Start in manual mode; still loads checkpoint settings")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--validate-frames", type=int, default=0, help="Auto-close after N GUI frames (validation only)")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    if args.run_dir:
        choices = checkpoint_choices(args.run_dir)
        if not choices:
            parser.error("No checkpoints in --run-dir yet; wait for the first save")
        args.checkpoint = str(max(choices, key=lambda p: p.stat().st_mtime_ns))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    try:
        session = NavigationPlayback(args.checkpoint, seed=args.seed, obstacle_count=args.obstacles)
    except ValueError as error:
        parser.error(str(error))
    if args.manual:
        session.mode = "manual"
    app = NavigationViewer(session, args.validate_frames, args.report, args.seed)
    app.watch.set(args.follow)
    app.root.mainloop()


if __name__ == "__main__":
    main()
