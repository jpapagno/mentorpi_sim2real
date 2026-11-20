# MentorPi Sim-to-Real Robotics

I'm teaching a Hiwonder MentorPi M1 to reach a goal and avoid obstacles using its onboard LiDAR. I train the policy in simulation, then run it on the robot's Raspberry Pi to see where it carries over and where it breaks down.

## Latest deployment · November 20, 2026

https://github.com/user-attachments/assets/f4524c2c-7724-4369-8b3e-42cf83e94292

The real MentorPi hesitates, then makes it through the cardboard doorway. Beside it, the same **7.15M TD3 checkpoint** drives through a generated doorway in the 2D simulator. The real footage plays at 2× and the simulated replay at 1×. [Full robot take at 1×](docs/media/deployment/2026-11-20/doorway-success.mp4?raw=true) · [Download the comparison](docs/media/simulation/2026-11-20/doorway-sim-to-real.mp4?raw=true)

## How the project evolved

### 1. From Isaac Lab to the first real goal

I brought the MentorPi URDF into Isaac Lab and started with a simple question. Could the robot learn to reach a position while avoiding obstacles? I trained an occupancy-grid policy with demonstrations, DAgger, and PPO. The Isaac demo showed the learned policy weaving through seven obstacles and reaching the goal in 14.15 simulated seconds. [Original Isaac demo](docs/media/mentorpi_goal_demo.mp4?raw=true)

On the Pi, I built the grid from the current LiDAR scan and used LiDAR odometry with IMU-assisted orientation to measure progress. The robot got around a bottle and reached the goal in Trial 1. [All four original deployment recordings](docs/results.md#onboard-measurements-and-limitations)

<details>
<summary>Watch the original bottle run</summary>

https://github.com/user-attachments/assets/e4f92932-6ad6-46e5-9340-1a3fcb0a9c69

</details>

### 2. Build a simulator around the task

The first task needed fast rollouts of planar motion, LiDAR returns, and collisions. I moved training into a lightweight Python and PyTorch simulator and used the same rectangle and polygon geometry for raycasts and collision checks. I also compared simulated scans with a stationary real scan to see how occlusion and missing returns changed the grid. [LiDAR comparison](examples/lidar_sim2real_improvements/) · [2D simulator](docs/planar.md)

The original deployed policy reached the goal in 150 of 150 simulated episodes in one fixed box-and-doorway layout, and in 43 of 150 episodes in randomized rooms. That gap made varied layouts a priority for the next training run. [Mixed-shape results](docs/results.md#lightweight-2d-mixed-shape-baseline--november-15-2026)

### 3. Simplify what the policy sees and does

I adapted Reinis Cimurs's TD3 navigation implementation and replaced the grid with the nearest valid LiDAR return in each of 36 sectors. Adding goal distance, goal bearing, and the previous action gives the policy 41 inputs. I kept the MentorPi's ability to strafe and held commanded yaw at zero so I could focus on translation. [TD3 reference implementation](https://github.com/reiniscimurs/DRL-robot-navigation) · [Policy interface](docs/planar.md#experimental-laser-sector-td3-policy)

On the floor, short commands could leave the robot visibly still. I measured how it started moving along each axis and used those measurements to model translation response. The next training run mixed open space, boxes, posts, rooms, and narrow doors. [Motion calibration](docs/calibration.md) · [Training scenario](scenarios/lidar_translation_varied_forever.json)

### 4. Revisit the doorway in simulation and on the floor

I replayed the **7.15M** checkpoint in two freshly recorded simulated doorway runs. It reached the goal in **8.05 s** in one layout and timed out at **60 s** in the other. Both runs used the narrow-door scene family from training. [Simulated goal arrival](docs/media/simulation/2026-11-20/doorway-goal.mp4?raw=true) · [Simulated timeout](docs/media/simulation/2026-11-20/doorway-timeout.mp4?raw=true)

The November 20 hardware session produced the hesitant success above, a success with cord contact, three collision clips, a hardware failure, and a back-and-forth failure. My next milestone is repeatable passage through the same doorway. [Success with cord contact](docs/media/deployment/2026-11-20/goal-with-cord-contact.mp4?raw=true) · [Collision example](docs/media/deployment/2026-11-20/collision-1.mp4?raw=true) · [Back-and-forth failure](docs/media/deployment/2026-11-20/back-and-forth.mp4?raw=true) · [All seven takes and replay details](docs/results.md#doorway-replays-and-real-deployment--november-20-2026)

## Explore the project

[Simulation and training](docs/simulation.md) · [Robot deployment](docs/deployment.md) · [Results](docs/results.md) · [Task roadmap](docs/roadmap.md)

Obstacle navigation is the first task. The roadmap covers changing surfaces, slip recovery, payload adaptation, moving obstacles, and pose-accurate docking.
