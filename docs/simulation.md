# Simulation and policy training

Task 01 trains the MentorPi to reach a position while avoiding obstacles. The environment is an Isaac Lab `DirectRLEnv`, with shared analytic dynamics in `mentorpi_nav/core.py` and RSL-RL policy training. The robot visual is optional for headless runs.

## Environment

Use the project's Isaac Lab 3 / Newton environment with PyTorch, TensorDict and RSL-RL available. Install this checkout with `python -m pip install -e .` or run modules from its root. Copy `local_paths.example.json` to ignored `local_paths.json` when supplying local simulator and robot asset paths. Robot assets, simulator installations and pretrained weights are external.

## Commands

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

Simulation grids come from known obstacle geometry. Real grids come from current LiDAR rays, with unobserved space occupied. This difference in visibility is part of the sim-to-real problem; the simulator does not render LiDAR or camera images as actor inputs.

Keep checkpoint files with their `task.json` and agent manifests. Record scenario settings, model architecture, seeds and checkpoint identity with each comparison. Goal-conditioned residual policies combine a goal-steering prior with learned obstacle corrections. A checkpoint trained without obstacles is not evidence of obstacle avoidance.

## Dynamics and evidence

The current environment integrates planar motion and uses swept-disc collision checks. Optional `mecanum_v1` dynamics share robot wheel mapping and apply wheel/body limits and response effects; they do not simulate individual roller contacts or measured surface friction. Isaac rendering provides visualization rather than proof of contact realism.

Scenario files under `scenarios/` define obstacle layouts and dynamics variation. Evaluate full episodes, report success/collision/timeout counts, and separate direct-goal cases from path-subgoal cases. [Results](results.md) summarize the recorded experiments; [the roadmap](roadmap.md) describes planned extensions.

For code changes, run the relevant existing checks with `python -m unittest discover -s tests -q` in the configured environment.
