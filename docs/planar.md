# Lightweight 2D LiDAR navigation

## Experimental laser-sector TD3 policy

The versioned `laser_sectors_36_goal2_action3_v1` interface is separate from
the deployed occupancy actor. It uses 41 floats: 36 minimum endpoint distances
in body-frame angular sectors, goal distance and bearing, then previous action
vx/vy/yaw. Sectors run counterclockwise from -pi; forward is sector18, left27,
right9. Distances are clipped at2.5m and divided by2.5; a sector without valid
returns is0 (unknown/blocked). Goal distance is divided by4*sqrt(2), bearing
by pi. Sensor endpoints are transformed into the body frame before binning.
Simulation uses the same 720-ray neighbor-min normalization as the robot scan.
The actor does not receive a grid, map or measured velocity. Drivetrain and
runtime progress guards still use measured motion. Outputs are three normalized
body velocities scaled by[.2,.2,.5].

This adapts [Cimurs's tutorial implementation](https://github.com/reiniscimurs/DRL-robot-navigation):
TD3 with two critics, delayed actor/target updates, target-policy smoothing,
CPU replay storage and a256x256 MLP. The simulator retains the10x8inch rectangle,
1.7m square arena and current-scan-only sensor noise. Rewards override to+100
on arrival or-100 on collision. Otherwise reward is0.5 times measured velocity
toward the goal (normalized by0.2m/s and clipped to[-1,1]), minus0.5 times
absolute normalized yaw velocity, minus0.5 times the existing linear footprint
proximity cost within0.2m. This uses motion toward the goal instead of the
tutorial's forward command so strafing can earn reward; the proximity threshold
is adapted to the smaller arena. This is not an exact reproduction or evidence
of successful learning. Other progress/time/action-smoothing coefficients in
the shared configuration do not apply in `reward_mode=td3_mecanum`.

```powershell
# New policy from random weights; no demonstration collection.
& "$env:USERPROFILE\Desktop\mentorpi\IsaacLab\.venv\Scripts\python.exe" -m mentorpi_nav.cli.td3 --output logs/laser_sector_td3_01 --device cuda:0
.\scripts\view-navigation.ps1 --run-dir logs/laser_sector_td3_01 --follow --auto-reset
```

The budget is2million total transitions across16environments, with10,000
initial random transitions, exploration noise decaying0.5 to0.1 over500,000
transitions and two gradient updates per vector step. Gamma=.995,tau=.005,
Adam lr=.0003,batch256,replay200,000,target noise=.2 clipped=.5,actor delay2.
Every25,000 transitions a stable checkpoint is saved; names count transitions,
not PPO iterations. `status.json` and `metrics.jsonl` retain exact completed
training episode counts, recent outcomes, exploration noise, device and throughput.
Timeouts bootstrap from their true final observation; arrival/collision do not.
Replay is not included in checkpoints: playback/evaluation/export are supported,
but this CLI intentionally starts fresh and does not claim exact resume support.
Use the existing `cli.evaluate` and `cli.export` entry points with these checkpoints.
Export validates actor/TorchScript parity and writes a distinct sector contract.
The robot runtime source selects this interface by contract; the existing Pi
installation is not changed by training or code edits.

The active navigation task can run and train without Isaac, USD, Newton, a
robot mesh or a graphics window. `NavigationState` owns analytic motion,
geometry, sensing, observations and rewards. The planar and Isaac adapters
share that core; Isaac remains available for optional 3D visualization and
future validated physical-dynamics work.

## Install and run

Use an environment with PyTorch, Gymnasium 1.x, TensorDict, Pillow and the
RSL-RL 5.5 API. The project's configured Isaac Lab interpreter already has
these dependencies; using that interpreter does not import the simulator.
For a separate environment, install this checkout with `python -m pip install
-e ".[planar]"`. The robot and SSH extras remain separate.

From the repository root:

```bash
# Fixed rectangular box, doorway, concave obstacle and thin round post.
python -m mentorpi_nav.cli.planar --mode smoke --scenario-file scenarios/lidar_box_doorway.json --num-envs 4 --steps 100 --output logs/planar_box

# Randomized rooms, rectangles, L-shaped polygons, doorways and posts.
python -m mentorpi_nav.cli.planar --mode smoke --scenario-file scenarios/lidar_rooms.json --num-envs 4 --steps 100 --output logs/planar_rooms

# Fine-tune a copy of an existing policy; use a fresh output directory.
python -m mentorpi_nav.cli.planar --mode train --scenario-file scenarios/lidar_rooms.json --checkpoint path/to/model.pt --transfer-scenario --num-envs 16 --device cuda:0 --iterations 500 --output logs/planar_training

# Existing complete-episode evaluator works on the new scenes too.
python -m mentorpi_nav.cli.evaluate --checkpoint path/to/model.pt --scenario-file scenarios/lidar_rooms.json --seeds 456 789 2026 --episodes 200 --output logs/planar_evaluation
```

Smoke runs use random actions unless given a checkpoint; their counts are not
policy benchmarks. They save `scene.png`, showing world geometry and rays next
to the exact policy grid. Training preserves the source checkpoint and records
the full scenario, architecture and source identity in its task/agent manifests.
`--transfer-scenario` resets optimizer and iteration state. A transfer retains
the existing shape, normalization, action scale and timing contract. This
command does not install a policy on the robot.

## Watch and inspect

`--network cnn_v1` starts a randomly initialized CNN without the residual
goal-seeking prior when no checkpoint is supplied. PPO's per-step discount is
configurable with `--gamma` (default 0.99), recorded in `agent.json`.
`--entropy-coef` controls PPO's exploration bonus (default 0.001). A higher
coefficient is an experiment to discourage early exploration collapse, not a
guarantee of successful navigation; change it separately from task rewards.
`scenarios/lidar_rooms_arrival50.json` is an experimental variant of the room
scenario with arrival reward 50 rather than 20; collision penalty remains 75.
For changed rewards, use a fresh output directory and `--transfer-scenario`
with the source checkpoint: actor and critic weights are retained, while the
optimizer and iteration counter restart. This is not a new from-scratch run
and has no established performance benefit yet.

```powershell
# Manual scene; W/S forward/back, A/D strafe, Q/E turn.
.\scripts\view-navigation.ps1 --manual --scenario-file scenarios/lidar_box_doorway.json --auto-reset

# Saved policy, automatically loading stable new checkpoints.
.\scripts\view-navigation.ps1 --run-dir logs/planar_training --follow --auto-reset

# Deliberately challenge an existing policy on a different scene.
.\scripts\view-navigation.ps1 --checkpoint logs/robot_shape_ppo/model_499.pt --scenario-file scenarios/lidar_rooms.json --auto-reset
```

The viewer shows actual circle and polygon outlines, sampled first-return
LiDAR rays, the occupancy grid, goal, trail and termination reason. Pause and
step to inspect a failure, or enable automatic respawn. Checkpoint views default
to the checkpoint's saved scene settings. An explicit `--scenario-file` runs a
labeled challenge scene while retaining the original training configuration;
it cannot change policy normalization, action scales or control timing. Use a manual scene to inspect geometry
before a corresponding policy exists. A saved checkpoint can also be viewed
through the retained Isaac evaluation adapter as described in
[the simulation guide](simulation.md#watching-checkpoints-during-training).

## Shared scene and geometry contract

`NavigationCfg.scene_spec` contains the portable scene definition and is
embedded in checkpoint manifests. Older scenarios and checkpoints default to
the original circle-only sampler and remain supported.

A fixed scene specifies start, goal, yaw and shapes:

- Circle: `type`, `center` in metres and positive `radius`.
- Rectangle: `type`, `center`, positive xy `size`, and optional `yaw` in radians.
- Polygon: `type` and 3–8 boundary-ordered world xy `vertices`.

Rectangles are represented by their four actual corners, not unions of circles.
Simple concave polygons are supported; degenerate and self-intersecting
polygons are rejected. Polygon and circle capacities are explicit task fields.
Room scene parameters control door placement/width, box dimensions/counts,
concave outline counts and post dimensions/counts. Reset sampling checks start
and goal clearance and a reachable route using the same geometry.

`geometry.py` provides polygon ray intersections, signed point distances and
continuous swept-disc collision checks. The CPU training teacher uses matching
segment and polygon tests. LiDAR sees the nearest actual shape edge or circle
surface and uses the existing scan normalization/rasterization. Unknown stays
blocked. The polygon geometry debug raster is conservative at cell corners;
the active LiDAR input uses exact edge intersections.

The default robot is a disc with analytic planar/mecanum response. Optional
`robot_size: [length, width]` enables an oriented rectangle for collision and
clearance rewards. Its `robot_radius` must enclose the rectangle and is used
conservatively by reset/path sampling. The rectangle sweep checks translation
and rotation with nine poses and a conservative inter-pose motion bound;
it does not rely only on the final pose. Body axes,
grid orientation, `[B,1,100,100]` occupancy, `[B,8]` state, speed scales and scan
timing are unchanged. This is not a contact, slip or optical-material simulation.
The box scene is a controlled geometry example, not a fitted reconstruction of
the recorded room.

### Small arena and rectangular body

`scenarios/lidar_small_rectangle.json` uses a 1.7 x 1.7 m arena and a
0.254 x 0.2032 m (10 x 8 inch) rectangular body, length along body +x.
The grid remains 100 x 100 at 0.05 m/cell (5 x 5 m), rebuilt from the current
scan. No accumulated map is added. The arena diagonal is under 2.5 m, so
every robot-to-goal vector fits inside the grid at any heading; occluded goals
are allowed. Noise, finite rays and occlusion still leave unknown cells.
Goal normalization is explicitly held at the original 4*sqrt(2) metres,
independent of the smaller arena. Existing checkpoints keep their saved
geometry and normalization. This scenario requires a new training run.
Room clutter is reduced to fit the arena: doorway, optional box and post.
Arrival reward remains 50, collision penalty 75; use `--gamma 0.995` explicitly.

```powershell
.\scripts\view-navigation.ps1 --manual --scenario-file scenarios/lidar_small_rectangle.json --auto-reset
```

## Environment adapters and the return to Isaac

- `PlanarGymEnv` is a standard single-environment Gymnasium API: NumPy dict
  observations and separate termination/truncation flags, with explicit reset.
- `PlanarBatch` keeps batched motion/sensing tensors on CPU or GPU.
- `RslPlanarEnv` maps that batch to the RSL-RL VecEnv protocol, including
  autoreset, outcome logging and timeout bootstrapping flags. No grids need to
  cross to CPU for training.
- `MentorPiEnv` remains the Isaac adapter. Its optional USD visualization uses
  the same polygons as extruded meshes. Those meshes do not add physical
  contact dynamics; headless stepping still uses the shared core.

The policy is independent of the environment adapter. A future physical Isaac
backend can provide measured pose/velocity and sensor rays through the same
observation processing, but new contact dynamics will require hardware
calibration and likely additional training. Interface compatibility alone does
not establish physical transfer.

When comparing checkpoints trained on different scenarios, challenge files
must explicitly reset inherited settings that define the comparison. In
particular, `original_four.json` clears polygon scenes and selects geometry
observations with body dynamics. The fixed LiDAR scene clears per-episode
noise ranges before setting its fixed noise. An omitted field otherwise keeps
the source checkpoint's value. Compare the saved full `navigation` objects,
not only the challenge filenames.

## Validation scope

Focused checks cover flat rectangle returns, concave free pockets, swept
collision/tunneling, teacher routes, partial geometry resets, randomized
reachable rooms, Gymnasium checks and adapter outcome semantics. Run:

```bash
python -m unittest discover -s tests -q
```

Only smoke training is used to verify the new execution path. No claim of
improved navigation success, new hardware deployment or physical realism is
made from these implementation checks.

The implementation was also compared with the headless Isaac adapter for 100
steps across two environments, including four collision/reset events. Poses,
observations, rewards and termination flags matched. This checks adapter parity
for the analytic task; it is not a comparison with contact simulation.
