# Simulation and onboard measurements

## Cord-clear motion pulses — November 16, 2026

Supervised short pulses bypassed the actor while retaining the motor guard,
sensor freshness and explicit disarming. At 0.06 m/s translation and 0.2 rad/s
yaw, 1.5-second pulses produced essentially stationary measurements; the
operator confirmed no visible motion. These samples are not valid gain fits.
At 0.1 m/s translation and 0.3 rad/s yaw, two one-second pulses per direction
produced measurable motion. Against the integrated logged applied command,
including command ramp and one second of settling, the observed response was:

| Direction | Measured displacement/angle | Measured / integrated command |
| --- | --- | --- |
| Forward | 7.1–7.5 cm | 84–87% |
| Backward | 6.8–7.4 cm | 79–88% |
| Left | 6.8–7.0 cm | 80–82% |
| Right | 5.8–6.3 cm | 67–75% |
| Counterclockwise | 7.2–7.8 degrees | 48–52% |
| Clockwise | 3.0–3.7 degrees | 20–25% |

Laser odometry and gyro agreed on these angles, but subsequent counterbalanced
tests showed that the apparent directional asymmetry was confounded by pulse
order. These are
short-pulse response measurements on one surface, not steady-state friction
coefficients or a full drivetrain identification. Sensor latency, motor
startup and stopping contribute to the ratios. Low-speed nonresponse and
history-dependent startup are not represented by the default ideal mecanum model;
a single global velocity scale is insufficient to describe this evidence.
No actuator gains, limits or trained checkpoints were changed by these tests.

With 1.5-second applied yaw pulses at 0.3 rad/s, reversing direction produced
about 10 degrees on the second pulse; repeating the same direction produced
about 20–21 degrees. The first pulse produced about 14–16 degrees in either
direction. A signed-memory response model fitted on takes 10 and 12 predicted
four held-out pulse angles in mirrored takes 11 and 13 with maximum error
1.08 degrees versus bias-corrected gyro integration. This is a narrow yaw
validation on one surface and voltage condition, not full drivetrain calibration.
See [calibration](calibration.md) for assumptions and unresolved stop buzzing.

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

## Lightweight 2D mixed-shape baseline — November 15, 2026

The original deployed `shape_499` actor was frozen and evaluated without Isaac
on the new rectangle/polygon scenes. These are direct-goal evaluations, without
teacher actions, global subgoals or a collision override. Each scenario used 50
complete episodes per seed on fixed screening seeds 456, 789 and 2026.

| Scenario | Success | Collision | Timeout | Success rate |
| --- | ---: | ---: | ---: | ---: |
| Fixed box and doorway | 150 | 0 | 0 | 100.0% |
| Randomized mixed-shape rooms | 43 | 104 | 3 | 28.7% |
| Original clean four-obstacle scenario | 147 | 3 | 0 | 98.0% |

The fixed box scene repeats one geometry with stochastic LiDAR inputs; it is
not 150 distinct layouts or evidence of general rectangle avoidance. Room
scenes contain variable rectangles, divider walls/doorways, concave outlines
and thin posts. Conditions differ in geometry, perception and drivetrain
response, so this is not an isolated shape or noise ablation.

Illustrative single-environment replays showed the actor going around the box
and through the doorway, a collision near a thin post in one room, and a
collision near a concave obstacle after local looping in another. These clips
are examples, not aggregate episode-index traces. Sensor RNG draw order differs
with batch size. No new training or hardware deployment was performed for this
baseline. Checkpoint identity, scenario settings, full episode records and
replays are retained in the ignored local experiment directory
`logs/planar_baseline_20261115_01/`.
