# Agent instructions: MentorPi sim-to-real robotics

## Project and code ownership

This repository develops learned control tasks for the Hiwonder MentorPi M1 mecanum platform, from simulation and policy training to ROS2 execution on Raspberry Pi. Goal-conditioned obstacle navigation is the first task; additional tasks are described in [the roadmap](docs/roadmap.md).

Keep changes in their owning module:

- `mentorpi_nav/core.py`, `drivetrain.py`: shared motion, coordinate and observation semantics.
- `mentorpi_nav/lab_env.py`, `agent.py`, `networks.py`, `teacher.py`: Isaac Lab environment, policy architecture and training-only demonstrations.
- `mentorpi_nav/cli/`: simulation, training, evaluation, playback and export entry points.
- `mentorpi_nav/robot/`: LiDAR preprocessing, measured odometry, actor execution, serial motor bridge and service lifecycle.
- `mentorpi_nav/media/`, `tools/`: demonstration recording and SSH transport.
- `deployment/`, `scenarios/`, `tests/`: active ROS configuration, experiment definitions and behavioral checks.

Use importable modules and `python -m` commands. Keep Windows wrappers in `scripts/` thin. Hardware access and executable work belong in explicit entry points, never at import time.

## Sim-to-real contracts

- Body coordinates are +x forward, +y left, +yaw counterclockwise; use metres, seconds and radians internally. Keep world, odometry, body and sensor frames explicit.
- The actor receives `[B, 1, 100, 100]` binary occupancy at 0.05 m per cell and `[B, 8]` state: body-relative goal (2), measured body velocity (3), previous action (3). Preserve grid axis order and normalization in `robot/navigation_math.py` and `core.py` together.
- The real obstacle input is the current LiDAR scan. The local grid is rebuilt from observed rays; unobserved cells remain occupied. Camera depth and a saved map are not dependencies of the current task.
- Relative goals are anchored once in odometry coordinates. Track progress from measured LiDAR odometry, with IMU-assisted orientation; commanded velocity is not measured motion. Current arrival checks position within 0.15 m and impose no final heading.
- Actor outputs are body velocities, converted to wheel commands by the shared drivetrain code. Motor channel order is front-left, rear-left, rear-right, front-right; forward signs are `(-1, -1, +1, +1)`. Diagnose logs and frame conventions before changing signs or scales.
- Treat observation shape, normalization, action scale, timing and checkpoint metadata as one interface. Changes require matching training/export/runtime updates and a numerical parity check where applicable.

## Simulation and experimental work

The current simulator uses analytic planar motion, including an optional mecanum response model; the visible robot mesh does not establish wheel-ground contact realism. Friction, payload, slip and surface-transition studies require an explicit dynamics model or contact simulation before making physical claims.

Keep task settings and seeds in scenario/manifests rather than hard-coded experiment branches. Distinguish direct-goal evaluation from global-subgoal comparisons, and held-out evaluation from training demonstrations. Retain checkpoint/task/agent provenance when exporting or comparing policies.

Prefer changes that improve the active task over extra frameworks, duplicate launch paths or one-off probes. Reuse existing checks; add a focused test when a contract or behavior changes. Do not create acceptance systems or elaborate test infrastructure without a concrete need.

## Validation and robot operation

Run relevant existing checks in the appropriate environment. The main suite is `python -m unittest discover -s tests -q`; simulation/training needs the configured Isaac Lab environment, and ROS execution needs the robot container. Use CLI `--help` for moved entry points. Report what was checked and what could not be exercised.

Code edits and local tests do not authorize physical motion. Run hardware commands only within the user's requested operating scope. Preserve explicit stop handling, command expiry, sensor freshness and documented actuator limits. The current deployment leaves obstacle avoidance to the actor; changing that control architecture must be deliberate and documented.

## Documentation and evidence

Keep the README focused on demonstrations, the first task and the task roadmap. Put simulation commands, deployment details and benchmark evidence in their corresponding `docs/` files. Describe demonstrated behavior confidently and precisely; label planned capabilities and distinguish operator-confirmed videos from aggregate hardware benchmarks.

For runtime changes, read [deployment](docs/deployment.md). For training or evaluation, read [simulation](docs/simulation.md) and [results](docs/results.md). For a new robot task, update [the roadmap](docs/roadmap.md) with its observation/action interface and measurable objective.

Keep credentials, machine paths, checkpoints, generated runs and private hardware-session notes out of commits. Repository cleanup must preserve model assets and recordings unless removal is requested.
