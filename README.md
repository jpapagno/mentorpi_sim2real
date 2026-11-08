# MentorPi Sim-to-Real Navigation

Learning-based goal reaching and obstacle avoidance for the Hiwonder MentorPi M1, with Isaac Lab training and evaluation.

## What this project does

- Trains a CNN navigation policy with behavior cloning, DAgger, and PPO.
- Uses a robot-aligned 100 x 100 binary occupancy grid plus relative goal, velocity, and previous action. Commands are forward, lateral, and angular velocity.

The simulation uses analytic planar motion and swept-disc collision checks inside an Isaac Lab DirectRLEnv. It does not model wheel contact physics. Training observations use ground-truth obstacle geometry; onboard occupancy comes from perception and mapping.

## Results and current status

A selected policy achieved **97.5-98% goal-reaching success across 600 simulated episodes** in the original four-obstacle benchmark (200 episodes each on three held-out seeds). A later deployment-oriented model achieved **98-99.5%** in a separate compound-obstacle simulation benchmark using global-path subgoals. These are separate experiments.

On Raspberry Pi 5, the later TorchScript policy had **5.20 ms p95 inference latency** over 300 calls under the running stack. Numerical export parity, live mapping/localization, and stationary watchdog behavior were checked. **Physical autonomous navigation is still being commissioned; no completed 30-goal hardware trial is claimed.**

This public repository contains the simulation, training, playback, and evaluation code. Robot deployment and hardware-control files are maintained separately and are not included.

See [benchmark results](RESULTS.md) and [historical experiment counts](navigation_goal_results.csv).

## Repository layout

| Path | Purpose |
| --- | --- |
| `mentorpi_nav/` | Navigation dynamics, observations, networks, training environment, teacher, evaluation, and playback |
| `scenarios/` | Obstacle layouts and motion uncertainty configurations |
| `tests/` | Navigation behavior, scenario, model-contract, and playback regression tests |
| `SIMULATION.md` | Detailed simulation interface and experiment commands |

## Setup and usage

This is a research workspace, not a standalone packaged application. Training uses the Isaac Lab 3 / Newton API installed for the original experiments, along with PyTorch, TensorDict, and RSL-RL. The robot mesh/USD assets, simulator installation, trained checkpoints, maps, recordings, credentials, and generated logs are not included.

Use your Isaac Lab Python environment. Copy `local_paths.example.json` to `local_paths.json` and set your local paths if using robot visuals. The USD asset is optional for headless training. PowerShell wrappers accept a local Isaac Lab root; Python entry points can also be called directly.

```powershell
# Check the simulation environment.
.\run-navigation.ps1 -LabRoot 'C:\path\to\IsaacLab' -TaskArgs @('--mode','smoke','--num-envs','4','--steps','300')

# Train an initial four-obstacle policy.
.\run-navigation.ps1 -LabRoot 'C:\path\to\IsaacLab' -TaskArgs @('--mode','train','--obstacles','4','--num-envs','8','--iterations','500','--log-dir','logs/grid_obstacles')

# Inspect a checkpoint you have trained.
python view_navigation.py --checkpoint logs/grid_obstacles/final.pt

# Evaluate a checkpoint on complete episodes.
python evaluate_suite.py --checkpoint logs/grid_obstacles/final.pt --scenario-file scenarios/original_four.json --seeds 456 789 2026 --episodes 200 --output logs/evaluation

# Run local regression tests in the configured Python environment.
python -m unittest discover -s tests -v
```

Checkpoints must retain their neighboring task/agent manifests. The example training command starts an experiment; reproducing the reported results requires the behavior-cloning/DAgger and PPO training stages, model architecture, scenario settings, and checkpoint selection described in the source and result notes. No pretrained weights are distributed here.

## References

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Hiwonder MentorPi ROS drivers](https://github.com/Hiwonder/MentorPi-ROS)
- [MentorPi asset/starter project](https://github.com/Matzefritz/HiWonder_MentorPi)
- [Nav2](https://github.com/ros-navigation/navigation2)

These external dependencies and assets are not bundled in this repository.
