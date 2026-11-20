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

Live LiDAR/IMU streams, policy inference, wheel translation and turning were checked. The operator confirmed goal arrivals in all four original recordings. [Trial 1](media/deployment/trial-1.mp4?raw=true) · [Trial 2](media/deployment/trial-2.mp4?raw=true) · [Trial 3](media/deployment/trial-3.mp4?raw=true) · [Trial 4](media/deployment/trial-4.mp4?raw=true)

The 30-goal autonomous hardware milestone is still open. Moving obstacles and reliable obstacle avoidance across layouts need dedicated physical trials.

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

## Doorway replays and real deployment · November 20, 2026

The two new simulation recordings use `model_7150000.pt` and its task
manifest. The saved actor ran a direct-goal policy on CPU without exploration
noise. Both replays use the `narrow_doors` entry as the scene override and
keep the checkpoint's sensor and motion settings. [Training scenario](../scenarios/lidar_translation_varied_forever.json)

Checkpoint SHA-256
`fe2e8ce0f2082851d0586803f20c6db89d56e1974529b2f0b44744333b3b6399`.

| Replay | Seed | Door width | Outcome | Simulated time | Final goal distance |
| --- | ---: | ---: | --- | ---: | ---: |
| [Doorway arrival](media/simulation/2026-11-20/doorway-goal.mp4?raw=true) | 2027 | 0.548 m | Reached the 0.15 m arrival radius | 8.05 s | 0.148 m |
| [Doorway timeout](media/simulation/2026-11-20/doorway-timeout.mp4?raw=true) | 456 | 0.534 m | Timed out | 60.00 s | 0.690 m |

These are selected replays from the training scene family. The timeout video
plays at 4× with a simulated clock. At the checkpoint save, the latest 100
simulated training episodes with exploration noise included 92 goal arrivals,
5 collisions, and 3 timeouts. The mean episode return was 92.76.

The checkpoint was deployed under the `7.15m` alias. The physical doorway was
roughly twice the robot's width. The [side-by-side video](media/simulation/2026-11-20/doorway-sim-to-real.mp4?raw=true)
pairs the hesitant success with seed 2027, an independently generated doorway.
The seven camera takes below are exploratory recordings, with outcomes labeled
from the original filenames.

| Recording | Original take |
| --- | --- |
| [Hesitant success](media/deployment/2026-11-20/doorway-success.mp4?raw=true) | `success_hesitent.MOV` |
| [Success with cord contact](media/deployment/2026-11-20/goal-with-cord-contact.mp4?raw=true) | `success_cord_hit_colision.MOV` |
| [Collision 1](media/deployment/2026-11-20/collision-1.mp4?raw=true) | `colision.MOV` |
| [Collision 2](media/deployment/2026-11-20/collision-2.mp4?raw=true) | `collision.MOV` |
| [Collision 3](media/deployment/2026-11-20/collision-3.mp4?raw=true) | `colision (2).MOV` |
| [Hardware failure](media/deployment/2026-11-20/hardware-failure.mp4?raw=true) | `hardware_fail.MOV` |
| [Back-and-forth failure](media/deployment/2026-11-20/back-and-forth.mp4?raw=true) | `back_and_forth_deployement_fail.mov` |

In a logged run from this phase, alternating commands produced about 2 mm of
net measured travel before the controller disarmed on its no-motion check.
The hardware session also included a connection interruption. The timing of
the last robot command during the interruption remains unverified.
