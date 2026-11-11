# MentorPi Sim-to-Real Robotics

Learning robot behaviors in simulation and transferring them to a physical Hiwonder MentorPi M1. The platform combines Isaac Lab policy training, LiDAR perception, onboard inference and mecanum drive control on Raspberry Pi.

**Task 01 is goal-conditioned navigation with obstacle avoidance.** It is the first step in a broader series of sim-to-real tasks covering changing surfaces, vehicle dynamics, perception and increasingly complex movement.

## 1. Simulation: learning to reach a goal

[![Learned policy navigating a seven-obstacle scene with its local occupancy grid](docs/media/mentorpi_goal_demo.gif)](docs/media/mentorpi_goal_demo.mp4)

[Watch the full-resolution simulation](docs/media/mentorpi_goal_demo.mp4)

The policy drives toward the green goal while observing a robot-relative occupancy grid. The display shows the route, local grid and remaining goal distance. This rollout reaches the 0.15 m arrival threshold in 14.15 simulated seconds; the animation plays at 2× speed.

Training uses behavior cloning, DAgger and PPO. The displayed simulation uses analytic planar dynamics inside Isaac Sim; wheel-ground contact and rendered sensors are not part of this experiment.

## 2. Real robot: around an obstacle

https://github.com/user-attachments/assets/e4f92932-6ad6-46e5-9340-1a3fcb0a9c69

**Trial 1:** the real MentorPi moves around the bottle and reaches its goal, as confirmed by the operator. The goal specifies position, so arrival does not require the robot to face forward.

[Download Trial 1](docs/media/deployment/trial-1.mp4) · [Additional recordings](docs/media/deployment/)

## Task 01: goal-conditioned obstacle navigation

The first task connects the complete learning-to-control loop:

```text
Simulation geometry / live LiDAR
                ↓
     100 × 100 local occupancy grid
                + goal, measured velocity, previous action
                ↓
         learned navigation policy
                ↓
       body velocity → wheel commands
```

On the robot, successive LiDAR scans estimate movement, and the IMU supports orientation estimation. The local obstacle grid is rebuilt as new scans arrive. No saved global map or camera depth is required.

Held-out simulation results include 97.5–98% goal-reaching success across three 200-episode four-obstacle evaluations. A separate global-subgoal experiment achieved 98–99.5% on compound-obstacle cases. These are simulation benchmarks; Trial 1 is a physical demonstration. [Methods and results](docs/results.md).

## Next tasks

| Task | What we want the robot to learn | Status |
| --- | --- | --- |
| **01 · Obstacle navigation** | Reach a selected position using local obstacle observations. | Initial simulation-to-hardware demonstration |
| **02 · Different-friction surfaces** | Drive across wood, tile, carpet and rubber, including transitions between surfaces. | Planned |
| **03 · Slip and disturbance recovery** | Recover from wheel slip, sideways drift and small external disturbances. | Planned |
| **04 · Payload adaptation** | Maintain motion accuracy as carried mass and load distribution change. | Planned |
| **05 · Constrained-space motion** | Strafe through narrow gaps, negotiate corners and park in tight spaces. | Planned |
| **06 · Moving obstacles** | Adjust the route or yield when obstacles enter the path. | Planned |
| **07 · Robust perception** | Handle LiDAR noise, occlusion, dropped returns and sensing delay. | Planned |
| **08 · Pose-accurate docking** | Reach a docking pose with both position and heading constraints. | Planned |
| **09 · Multi-goal missions** | Complete waypoint sequences while managing accumulated odometry error. | Planned |

Future tasks will extend the dynamics and sensing models as needed, then compare simulation and hardware behavior. In particular, friction and slip studies will add wheel-ground dynamics beyond the current analytic model. [Task roadmap](docs/roadmap.md).

## Run the project

Use the configured Isaac Lab Python environment for simulation and the configured ROS2 Humble container on the robot. Trained weights and third-party robot assets are not bundled.

```bash
# Inspect or evaluate a trained checkpoint.
python -m mentorpi_nav.cli.view --checkpoint path/to/final.pt
python -m mentorpi_nav.cli.evaluate --checkpoint path/to/final.pt --output logs/evaluation

# On the configured Pi, from the repository root: goal 1 metre forward.
sudo python3 -m mentorpi_nav.robot --goal-relative 1 0
```

## Documentation and code

| Resource | Contents |
| --- | --- |
| [Simulation](docs/simulation.md) | Training, evaluation, playback and policy input contract |
| [Robot deployment](docs/deployment.md) | ROS modules, installation, goals and motor interface |
| [Results](docs/results.md) | Benchmark counts, inference measurements and evidence |
| [Roadmap](docs/roadmap.md) | Planned robot tasks and experiment objectives |
| [Agent instructions](AGENTS.md) | Code ownership and sim-to-real engineering conventions |

Shared dynamics and policy code live in `mentorpi_nav/`; executable modules are grouped under `cli/`, `robot/`, `media/` and `tools/`. Windows launchers are in `scripts/`, robot configuration in `deployment/`, and task scenarios in `scenarios/`.
