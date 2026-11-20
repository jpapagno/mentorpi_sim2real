# Fixed-controller motion calibration

This workflow is separate from RL training and the installed navigation actor.
Its purpose is to measure command-to-motion response and validate short physical
targets while keeping the controller algorithm fixed. The current profile is
experimental; motor buzzing after motion is unresolved and closed-loop targets
have not yet been validated on hardware.

## Measurement and profile

### Translation-only identification, November 16

Three translation-only runs completed 72 one-second pulses at 0.06, 0.08 and
0.1 m/s, with repeated/reversed forward, backward, left and right motion. No
yaw was commanded. Runs 1/2 fit an empirical signed-memory and deadzone model;
run 3 reserved 24 pulses. The scalar applied-command replay had 2.46 mm RMS
distance error and 6.13 mm maximum. Shared-simulator replay of requested
commands, including ramps and explicit disarming during holds, had 4.72 mm RMS
and 8.60 mm maximum versus RF2O. Baseline was 14.34 mm RMS / 30.54 mm maximum.
Held-out gyro rotation was below 0.85 degrees. No capture aborted.

The response is implemented in `translation_response.py`, fitted through
`python -m mentorpi_nav.cli.fit_translation_history`, and verified through
`python -m mentorpi_nav.cli.validate_translation`. The scenario
`scenarios/lidar_sector_td3_translation_calibrated.json` embeds fitted parameters
and removes the old broad actuator scaling. It retains existing command ramps,
action delay and zero commanded yaw. Checkpoints retain task/profile metadata.

Scope is settled pure-axis motion on one surface against odometry. Independent
metric scale, longer durations, diagonals, voltage variation and startup/stop
timing are not validated. Ten-second memory decay and cold initial memory per
block are assumptions. The fitted memory is empirical, not a firmware diagnosis.
The post-command translation abort limit is now 5 cm, allowing earlier observed
estimator settling up to about 4 cm; the rotation limit remains 0.12 rad.

The earlier mixed-motion suite was aborted after a yaw failure; these new
captures contain no commanded turns. No motor gains or firmware were changed.

Use body coordinates: forward/left metres, counterclockwise radians. Translation
comes from measured LiDAR odometry; yaw is checked against bias-corrected gyro
integration. An IMU is not an accurate displacement sensor. Timestamped sensor
data and settled endpoint error must be distinguished from motor response.

`robot/calibrate_motion.py` records bounded pulses, commanded wheel/body output,
fused pose, laser pose, gyro rotation and settling. `--allow-stationary` is only
for short threshold tests: a stationary sample is recorded as such, not accepted
as a useful gain estimate. The probe stops after each pulse. A sequence's
`passed` flag means its collection protocol completed; inspect the measured
response before claiming calibration.

Fit a profile from one set of recordings and reserve separate recordings:

```powershell
python -m mentorpi_nav.cli.calibration --fit logs/fit_a.json logs/fit_b.json --validate logs/heldout_repeat.json --output outputs/calibration
```

The profile contains six signed command-response lookup tables with monotonic
response estimates from trailing 0.35-second pose slopes. Short windows are
sensitive to estimator lag and noise. The November16 exploratory fit uses36
pulses and reserves6 repeat pulses; errors are too large in some directions
to call this a complete calibration. Do not apply those lookup tables to the
navigation actor or treat them as identified steady-state motor parameters.

Counterbalanced yaw tests subsequently showed that these fixed-order lookup
tables are confounded by command history. They are superseded as a yaw model;
do not compensate the robot using separate fixed clockwise/counterclockwise gains.

## Experimental history-dependent yaw model

Repeating a 0.3 rad/s, 1.5-second turn produced roughly 20–21 degrees on the
second pulse; reversing produced roughly 10 degrees. First pulses in either
direction produced roughly 14–16 degrees. This supports actuator memory,
startup and reversal effects, but does not identify whether firmware control,
stiction, backlash or another mechanism causes them.

`drivetrain_response.py` implements signed command memory, a moving threshold,
rate rise and stop decay. The offline `python -m mentorpi_nav.cli.fit_response`
entry point fits timestamp-aligned applied-command and raw-gyro recordings.
Takes 10/12 were fitted; mirrored takes 11/13 were held out. Their four pulse
angle prediction errors were at most 1.08 degrees. Minimum moving command and
10-second idle-memory decay were assumed, not identified; memory starts cold
for each capture. This fit covers 0.3 rad/s pulses on one surface, not mixed
motion, arbitrary speeds, battery variation or a physical explanation of buzz.

The optional scenario `scenarios/lidar_sector_td3_yaw_identified.json` uses
this response with hardware command caps. Defaults and the deployed actor are
unchanged. Scalar/batched numerical parity and selective reset are tested;
four environments completed a 100-step CPU smoke run. Existing scenario action
delay remains an assumption and may overlap some fitted delay. Do not treat
this scenario as a fully identified dynamics model or start training before
broader motion validation establishes its useful range.

## Repeatable hardware characterization suite

`python -m mentorpi_nav.robot.calibration_suite --output unused` previews the
fixed protocol without ROS or motion. The explicit `--execute` mode is a Pi-host
supervisor for the existing deployed `calibrate_motion.py` ROS entry point.
It runs three identical captures by default, with 24 one-second pulses per run:
balanced longitudinal/lateral repeats and reversals, both yaw starting orders,
low-command thresholds, and a five-second idle reversal. No drivetrain settings
or firmware are changed. Service restart is recorded and is not a power reset.

Execution output must be a new capture directory below
`/home/pi/docker/tmp/mentorpi_nav`; that directory is mapped into the container.
Use the installed compatible runtime, fresh sensors, clear floor/cords and the
authorized hardware operating scope. Each block retains displacement/clearance
limits and explicit zero/disarm; its supervisor has a 55-second deadline.
Any failed block aborts the suite and stops the service. Low-command stationary
responses are deliberately recorded rather than counted as successful movement.

Each run contains raw ROS bags and per-block JSON traces. The manifest records
the ordered protocol and its SHA256. `summary.json` compares matching ordered
pulse slots across runs: sample count, mean, standard deviation, range and
coefficient of variation for body/LiDAR displacement, gyro angle, command
integral, gain and settling displacement. CV near zero mean is not useful;
inspect absolute spread and failed/missing trials. Recompute offline with
`python -m mentorpi_nav.robot.calibration_suite --summarize --output CAPTURE_DIR`.
The recordings do not identify audible buzz, motor current or actual wheel RPM.

## Fixed target controller

`robot/motion_profile.py` contains the same proportional law for both the
identity baseline and fitted profile: kp_xy=1, kp_yaw=1.5. Only the lookup-based
command compensation changes. It preserves current limits of0.1m/s per
translation axis and0.3rad/s yaw, caps compensation to the tested range and
stops within1cm position and2degrees heading error. Below a measured moving
command, it uses the smallest tested moving command, so arbitrarily slow
tracking and arbitrarily small errors are not guaranteed.

Preview a short target without ROS initialization or motor access:

```powershell
python -m mentorpi_nav.robot.track_motion --target 0.06 0 0 --profile outputs/calibration/profile.json --output logs/target_check.json
```

Physical execution requires `--execute` in the configured robot ROS environment.
This entry point is not installed by generating a profile and has not been
commissioned on the Pi. It requires fresh sensors, single policy ownership,
observed obstacle clearance, a10-second deadline, displacement limits,
no-measured-response stopping, explicit zero/disarm and a settled final check.
The fitted profile and controller remain diagnostic-only until the stop behavior
is understood and supervised validation is complete.

Validate both profiles with the exact same controller and held-out targets:
forward/backward, left/right, both yaw signs, then diagonals and combined motion.
Record endpoint position/heading error, repeated trials, overshoot, settling,
unintended axes, starting versus already-moving response and battery voltage.
Check sensor scale independently with a ruler/floor marks before declaring
physical accuracy. Do not fit and validate on the same target recordings.

## Stop/buzz investigation

The current zero-speed packet matches the installed vendor SDK byte-for-byte.
The recorded long post-turn hold contains disarmed, exact-zero wheel commands.
The operator heard buzzing, but gyro data showed essentially no sustained turn
after the initial stop transient. Board motor-current, PWM duty and actual wheel
speed are not reported by this runtime, so zero-command logs do not prove motor
torque release or identify the sound source. No undocumented brake/release
packets, firmware changes or gain increases were applied.

Before more physical tracking tests, identify whether the sound is from a wheel
motor, the board buzzer or another component. This requires direct observation;
the SDK and motion traces alone cannot establish it. Keep source-level stopping
behavior separate from hardware/firmware hypotheses. A manufacturer support
case involving an RRC Lite and zero-command stop problems exists, but it does
not establish the cause on this robot:
[Hiwonder support discussion](https://forum.hiwonder.com/t/mentor-pi-mechanum-chasis-motors-dont-stop-spinning/312).
