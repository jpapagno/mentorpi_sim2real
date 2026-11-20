# Simulation and policy training

Task 01 trains the MentorPi to reach a position while avoiding obstacles. The
active LiDAR navigation path uses a lightweight PyTorch/Gymnasium environment
with RSL-RL policy training. An Isaac Lab `DirectRLEnv` adapter remains available;
both share analytic dynamics, shape geometry and observations in
`mentorpi_nav/core.py`. Start with the [2D LiDAR guide](planar.md) for rectangular
boxes, polygons, room randomization and training without the simulator. The
commands below document the retained Isaac path and earlier experiments.

## Environment

An optional measured-yaw experiment is documented in [calibration](calibration.md).
Its scenario is `scenarios/lidar_sector_td3_yaw_identified.json`; it captures
short-turn command history but does not establish contact/friction realism or
a complete calibrated robot. It is not enabled in existing runs.

Use the project's Isaac Lab 3 / Newton environment with PyTorch, TensorDict and RSL-RL available. Install this checkout with `python -m pip install -e .` or run modules from its root. Copy `local_paths.example.json` to ignored `local_paths.json` when supplying local simulator and robot asset paths. Robot assets, simulator installations and pretrained weights are external.

## Commands

For indefinite translation-only TD3 training, use `--forever` and the mixed
scene scenario. `--init-checkpoint` restores actor, critics, targets and optimizer
state while starting a fresh replay buffer and new-run transition counter.
Source path, hash and source transition count are saved in `task.json`.
Mixture sampling retries up to ten batches of 100 candidates in the chosen
scene family without relaxing collision/reachability criteria. Invalid scene
configuration still fails. Unexpected errors inside the training loop write
`crash_TRANSITIONS.pt` before propagating the error. Recovery with
`--init-checkpoint` restores networks/optimizers but starts fresh replay and
episode/RNG state, so it is not an exact uninterrupted continuation. Always
use a new output directory; periodic retention does not delete crash checkpoints.

```powershell
& "$env:USERPROFILE\Desktop\mentorpi\IsaacLab\.venv\Scripts\python.exe" -m mentorpi_nav.cli.td3 --scenario-file scenarios/lidar_translation_varied_forever.json --init-checkpoint logs/translation_calibrated_td3_20261116_01/model_1300000.pt --output logs/translation_varied_forever_01 --device cuda:0 --gpu-memory-fraction 0.75 --num-envs 16 --forever --random-steps 0 --noise-start 0.2 --noise-end 0.1 --save-every 100000 --keep-checkpoints 20 --seed 73010
```

This runs until Ctrl+C, which writes an interruption checkpoint. Keep the laptop
awake. Periodic saves retain the latest 20 files; an interruption save can add
one file. There are no checkpoint evaluations or automatic restarts. Use a fresh
output directory for each invocation. `status.json` has null target/ETA in
continuous mode. Open, box, post/concave, room and narrow-door scenes are sampled
per episode. Noise remains 0.5–2 mm Gaussian range noise and 5–7% independent
ray dropout, with first-return occlusion and existing sensor preprocessing.
These are measured-reference augmentation settings, not a fully identified
material/motion-dependent optical model. Calibration retains its short pure-axis
validation limits. This command does not deploy to or move the robot.

Translation-only sector TD3 keeps the existing three-output contract but hard
masks yaw to zero in the actor, exploration, target smoothing and environment.
Checkpoint/task metadata must agree; playback and exported actors retain the mask.
The scenario retains the existing noise, rooms, reward and actuator scaling,
with 0.1 m/s translation caps. This is an experiment isolating commanded turns,
not a validated fix for the hardware stop/buzz failure or unintended yaw drift.

```powershell
python -m mentorpi_nav.cli.td3 --scenario-file scenarios/lidar_sector_td3_translation_only.json --output logs/translation_td3_new_run --device cuda:0 --gpu-memory-fraction 0.75 --num-envs 16 --transitions 2000000 --save-every 25000 --seed 73001
```

Use a fresh output directory. Read `status.json` for transitions, actual device,
throughput and recent outcomes; `metrics.jsonl` retains the history. Model saves
occur every 25,000 transitions. Compare direct-goal checkpoint evaluations before
selecting a policy for hardware. Training does not change the Pi installation.

```bash
# Start a short headless environment run.
python -m mentorpi_nav.cli.simulation --mode smoke --num-envs 4 --steps 300

# Train a four-obstacle policy.
python -m mentorpi_nav.cli.simulation --mode train --obstacles 4 --num-envs 8 --iterations 500 --log-dir logs/grid_obstacles

# Inspect a checkpoint, or drive manually without one.
python -m mentorpi_nav.cli.view --checkpoint logs/grid_obstacles/final.pt
python -m mentorpi_nav.cli.view --manual

# Evaluate complete episodes on fixed held-out seeds.
python -m mentorpi_nav.cli.evaluate --checkpoint path/to/final.pt --scenario-file scenarios/original_four.json --seeds 456 789 2026 --episodes 200 --output logs/evaluation

# Export the selected residual actor for robot inference.
python -m mentorpi_nav.cli.export --checkpoint path/to/final.pt --output outputs/robot_policy
```

`mentorpi_nav.cli.train` provides demonstration and DAgger training. `mentorpi_nav.cli.evaluate_local` compares actor control using global-path subgoals in simulation; it is separate from the robot's direct-goal runtime. Use each command's `--help` for its checkpoint and experiment arguments.

On Windows, `scripts/run-navigation.ps1` and `scripts/view-navigation.ps1` select the configured interpreter. `scripts/preview-navigation.bat` handles the standalone Kit/OpenUSD preview environment. Media commands are `python -m mentorpi_nav.media.record` and `python -m mentorpi_nav.media.compose`.

## Model contract

The policy reads a robot-aligned `[B,1,100,100]` binary occupancy grid at 0.05 m per cell and `[B,8]` state containing relative goal, measured body velocity and previous action. Actions are forward, lateral and angular velocity. Body axes are +x forward, +y left and positive counterclockwise yaw.

Existing scenarios default to complete obstacle geometry. `scenarios/robot_lidar.json`
instead uses analytic first-return LiDAR against the same circle-union obstacles
and arena walls used for collision checking. Only rays before the nearest hit
clear space; occluded and missing-ray cells remain occupied. The binary grid
shape and eight state values are unchanged. No camera or rendered sensor is needed.

The LiDAR scenario uses the stationary November 14 capture as a reference:
nominally 500 full-circle raw rays (including both angular endpoints), the recorded body
mount offset, the runtime's conservative 720-ray normalization, 5 cm ray
rasterization, and a one-cell endpoint uncertainty neighborhood. Scans update
every 0.1 s and are held between updates, including repeat observation calls.
Partial episode resets invalidate only the reset environment's scan.

The stable capture had approximately 6.24% invalid raw returns. The scenario's
independent 6% ray dropout and 2 mm Gaussian range noise are initial augmentation
settings, not a fitted optical noise model. The captured raw count varied from
498 to 501; the initial model holds it at 500. Jagged edges
also arise naturally from occlusion, finite rays, rasterization and compound
geometry. A stable stationary capture does not establish noise under motion,
material-dependent returns, or correlated dropout. Physics and collision geometry
remain unchanged; do not add artificial geometric edges merely to mimic clutter
in one room. Unknown is still encoded as blocked; separate hit/free channels
remain a later policy-interface experiment.

```bash
# Smoke-test the LiDAR scenario in Isaac Lab.
python -m mentorpi_nav.cli.simulation --mode smoke --scenario-file scenarios/robot_lidar.json --num-envs 4 --steps 100 --log-dir logs/lidar_smoke --visualizer none

# Compare an existing actor on LiDAR observations without retraining.
python -m mentorpi_nav.cli.evaluate --checkpoint path/to/final.pt --scenario-file scenarios/robot_lidar.json --seeds 456 789 2026 --episodes 200 --output logs/lidar_baseline
```

Scenario settings are saved in `task.json` and evaluation reports. Moving a PPO
checkpoint to these changed observations requires `--transfer-scenario` and a new
log directory; this resets optimizer/iteration state. No existing policy is
assumed to work on the changed observations without evaluation and retraining.

### Clutter and stronger sensor randomization

See the [LiDAR sim-to-real example](../examples/lidar_sim2real_improvements/README.md)
for the animated comparison with the stationary hardware capture.

`scenarios/robot_lidar_clutter.json` retains the original LiDAR scenario for
comparison and adds a harder training distribution. Each layout contains 36–64
circle pieces forming thin posts, uneven short faces and compound patches.
These are shared collision, teacher and sensor geometry, rather than noise
painted onto the occupancy grid. Reachable starts are selected 0.3–0.7 m from
the nearest obstacle surface (distance measured from robot center).

Sensor settings are sampled per episode: Gaussian range standard deviation
3–12 mm and independent ray dropout 3–10%. Three persistent angular dropout
sectors, each 1–5 degrees wide, are sampled at reset. Sector locations remain
fixed in the sensor frame for that episode. Each scan additionally perturbs
1% of ranges by a uniform signed error up to 8 cm. Noise and isolated dropout
are refreshed with the scan, not on repeat observation calls; partial resets
resample only the reset environments. Range limits still reject invalid returns.
The period stays at 10 Hz; timing jitter is not modeled.

These settings are robustness augmentations, not calibrated material-dependent
sensor effects. In particular, persistent sectors represent an episode-level
blind-angle stress case. They do not claim to model occlusion from a particular
object or the temporal correlations of the real sensor. More irregular scene
geometry creates most spatial boundary complexity; increasing white range noise
alone cannot reproduce a cluttered room. Keep both scenarios for comparisons.

```bash
python -m mentorpi_nav.cli.simulation --mode smoke --scenario-file scenarios/robot_lidar_clutter.json --num-envs 4 --steps 100 --log-dir logs/lidar_clutter_smoke --visualizer none
```

Keep checkpoint files with their `task.json` and agent manifests. Record scenario settings, model architecture, seeds and checkpoint identity with each comparison. Goal-conditioned residual policies combine a goal-steering prior with learned obstacle corrections. A checkpoint trained without obstacles is not evidence of obstacle avoidance.

## Watching checkpoints during training

The Isaac evaluation path can follow a training directory while the trainer
runs separately. Wait for `model_0.pt` to exist, then use the configured Kit
launcher on Windows (from the repository root):

```powershell
.\scripts\preview-navigation.bat --mode evaluate --checkpoint logs/lidar_noise_ab_20261115/current_noise/model_0.pt --follow --realtime --steps 0 --log-dir logs/watch_noise_current
```

`--steps 0` keeps the viewer running until its window closes. `--realtime`
paces the simulation for viewing. The viewer loads navigation settings from
the checkpoint's `task.json`, so the LiDAR/dynamics configuration matches its
training run. Arrivals, collisions and timeouts automatically reset the scene.
`--follow` waits for a new checkpoint's size and timestamp to stabilize, loads
it, and starts new episodes. Outcome counts are printed to the viewer log.
This is deterministic actor playback of saved weights, not the trainer's
stochastic rollouts. The visible mesh is still analytic navigation visualization,
not contact-based wheel physics.

For the milder-noise comparison, use
`logs/lidar_noise_ab_20261115/mild_noise/model_0.pt` and a separate
`--log-dir logs/watch_noise_mild`. That run begins after the first training run.
Close the viewer to release rendering resources; this does not stop training.
Do not write viewer logs into the training directory.

The lighter CPU viewer shows the scene and exact policy grid without the
Isaac renderer:

```powershell
.\scripts\view-navigation.ps1 --run-dir logs/lidar_noise_ab_20261115/current_noise --follow --auto-reset
```

Its Pause, Step, Auto-reset, speed and checkpoint controls allow inspection of
failures. Either viewer can be opened for each run once its first checkpoint
exists. Multiple 3D windows share GPU resources with training.

### Matched noise comparison

The November 15 experiment initializes both variants from the original
`shape_499` checkpoint and uses seed 41001, 1,000 PPO iterations, 16 environments,
32 rollout steps, fixed learning rate 1e-4, and entropy coefficient 0.001. Each
variant collects 512,000 transitions and saves every 100 iterations. The runs
are queued sequentially on the local GPU.

`robot_lidar_clutter_mild.json` keeps the clutter geometry, dynamics, mounting,
timing and reward settings identical to `robot_lidar_clutter.json`. Only sensor
degradation changes: Gaussian range standard deviation 0.5–2 mm, independent
dropout 5–7%, no persistent blind sectors and no range outliers. This is a mild
test condition, not a calibrated optical sensor model.

The local worker records its state in
`logs/lidar_noise_ab_20261115/status.json`. After both training runs, it evaluates
checkpoints 100, 300, 600 and final on both noise settings, then additionally
evaluates each final actor on the original clean scenario. Screening uses fixed
seeds 456, 789 and 2026, 25 complete episodes per seed at intermediate checkpoints
and 50 at final. Summaries accumulate in `comparison.json`. Training reward and
recent reset metrics are not substitutes for these episode evaluations. No
hardware deployment is part of this experiment.

## Dynamics and evidence

The current environment integrates planar motion and uses swept-disc collision checks. Optional `mecanum_v1` dynamics share robot wheel mapping and apply wheel/body limits and response effects; they do not simulate individual roller contacts or measured surface friction. Isaac rendering provides visualization rather than proof of contact realism.

Scenario files under `scenarios/` define obstacle layouts and dynamics variation. Evaluate full episodes, report success/collision/timeout counts, and separate direct-goal cases from path-subgoal cases. [Results](results.md) summarize the recorded experiments; [the roadmap](roadmap.md) describes planned extensions.

For code changes, run the relevant existing checks with `python -m unittest discover -s tests -q` in the configured environment.
