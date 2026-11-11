# Simulation and onboard measurements

## Original four-obstacle benchmark

Selected residual CNN policy, initialized with demonstrations/DAgger and fine-tuned with PPO. Fixed 200 complete episodes per seed. Actor supplies the velocity actions; the A* teacher is used only for demonstration collection.

| Seed | Success | Collision | Timeout | Success rate |
| --- | ---: | ---: | ---: | ---: |
| 456 | 196 | 3 | 1 | 98.0% |
| 789 | 195 | 3 | 2 | 97.5% |
| 2026 | 195 | 4 | 1 | 97.5% |
| 2027 (additional) | 191 | 5 | 4 | 95.5% |

The three primary seeds passed the empirical target of >90% success and <=2% collisions individually. The additional seed had 2.5% collisions. Expanded varied-count, passage, and long-route challenges reached 84%, 90%, and 86% success, with 7-11% collision rates. These results do not establish universal collision avoidance.

## Later deployment-oriented policy

Trained with compound circle-union obstacle shapes, actuator scaling, and 100 ms action delay. Each entry below is success/collision/timeout over 200 episodes. The global-path experiment uses A* subgoals at 0.5 m lookahead and learned actor velocities, with no teacher actions or collision override.

| Seed | Original four, direct goal | Compound shapes/delay, direct goal | Compound shapes/delay, global-path subgoals |
| --- | --- | --- | --- |
| 456 | 197/1/2 | 182/15/3 | 196/4/0 |
| 789 | 197/2/1 | 180/17/3 | 199/0/1 |
| 2026 | 196/2/2 | 177/21/2 | 199/1/0 |

Global-path results are distinct from direct-goal performance and physical deployment acceptance.

## Onboard measurements and limitations

Raspberry Pi 5, TorchScript export of the later policy: maximum action difference approximately 1.20e-6 across 32 golden cases. Under the running stack, 300 inference calls gave p95 5.20 ms and maximum 7.19 ms.

Live LiDAR/IMU streams, policy inference, wheel translation and turning have been checked. Four recorded runs have operator-confirmed goal arrivals; the README features Trial 1. No successful autonomous 30-goal hardware session has been completed; sensor noise, moving obstacles, and reliable general obstacle avoidance are not established by these simulation results.

Counts are transcribed from local experiment records. Generated episode traces and checkpoints are excluded from this source publication. The [historical experiment CSV](../navigation_goal_results.csv) does not include every later deployment experiment.
