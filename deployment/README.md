# MentorPi M1 deployment

This directory is the Raspberry Pi deployment package for the Isaac Lab MentorPi Motion task. Run its Python commands from here. The runtime uses the MentorPi ROS 2 Humble container, NumPy, CPU PyTorch and pyserial. The training environment is only needed for export. The motor and sensor stack comes from the [MentorPi sim-to-real repository at `5bbe076`](https://github.com/jpapagno/mentorpi_sim2real/tree/5bbe076).

## Modules

| Module | Responsibility |
| --- | --- |
| `rrc_serial`, `drivetrain`, `hardware_bridge`, `command_guard` | Own the RRC Lite serial port, convert body velocity to wheel RPS, and stop wheel output when commands or sensor heartbeats expire. |
| `scan_normalizer`, `measured_odometry`, `navigation_io` | Read the live scan, obtain RF2O/EKF pose and measured velocity, and transform LiDAR points into the robot frame. |
| `policy_runtime`, `isaac_motion`, `policy` | Verify the bundle, translate robot observations and actor outputs, and run the goal loop with time, displacement and progress checks. |
| `calibrate_motion` | Send bounded diagnostic pulses through the same motor bridge. Record requested motion, guard diagnostics and laser odometry. |
| `launch`, `service`, `cli` | Start the disarmed ROS stack and provide Pi host start and stop commands. |

The motor bridge uses `/mentorpi/policy_cmd_vel` and `/mentorpi/motion_enable`. Motor IDs 1–4 are front-left, rear-left, rear-right and front-right. Forward motor signs are `(-1, -1, +1, +1)`. Wheel geometry and limits live in `config/drivetrain.json`. The serial device alias is defined in `rrc_serial.py`; check it against the Pi image.

## Export and select a policy

The local bundle `models/isaaclab-motion-900` was generated from training run `2026-11-25_02-02-04`, checkpoint `model_900.pt`. It contains `policy.ts`, `contract.json` and `parity_cases.npz`. Bundles live outside Git. Generate bundles from this directory with the Isaac Lab training Python environment active.

```bash
python -m mentorpi_deploy.export_isaac_motion \
  --checkpoint ../logs/rsl_rl/mentorpi_motion/2026-11-25_02-02-04/model_900.pt \
  --output models/isaaclab-motion-900
```

Use a new output directory name for each checkpoint. Export checks the saved training settings and runs the deterministic RSL-RL actor against the reloaded TorchScript actor. Verify an exported bundle on the workstation or the Pi before a live run.

```bash
python -m mentorpi_deploy.policy --policy isaaclab-motion-900 --verify-only
```

The Motion actor takes 80 floats, ordered as position XYZ, measured linear velocity XYZ, goal offset XY and 72 LiDAR ranges. The starting pose defines the policy world frame. Static height is 0.04 m and vertical velocity is zero. The ray order is −180° through 175° at 5° increments in that frame, with misses at 3 m. The hardware adapter bins transformed LiDAR returns conservatively. Scan mounting, height, occlusion, odometry drift and sensing cadence can differ from the simulated rays.

The actor outputs world-frame XY velocity with a nominal 60 Hz policy period. The adapter clips the normalized outputs, rotates the velocity into the robot body frame and commands zero yaw. The runtime scales the body command as needed to meet the 0.1 m/s per-axis limit while retaining direction. The motor bridge applies wheel and slew limits. Training used a simulated block with direct velocity control, so numerical parity alone cannot establish physical tracking or obstacle avoidance.

To deploy a checkpoint with a different observation or action contract, add a task adapter in `mentorpi_deploy`, register it in `policy_runtime.ADAPTERS`, and export matching metadata and parity cases. The current adapter interface accepts a single flat TorchScript input and output. The ROS stack always consumes the final body velocity tuple `(forward_mps, left_mps, yaw_radps)`.

## Pi setup and operation

Copy this directory, including the chosen bundle, to `/home/pi/docker/tmp/mentorpi_isaac_deploy`. The unit expects it at `/home/ubuntu/shared/mentorpi_isaac_deploy` in the existing `MentorPi` container. The container user needs serial and workspace write access. Verify the LiDAR, RF2O, robot description and IMU packages in the container and the device aliases in the deployment configuration.

From the deployment directory on the Pi host, stop the prior navigation unit if installed, then install the supplied unit under the same service name.

```bash
sudo systemctl stop mentorpi-navigation.service
sudo install -m 644 mentorpi-navigation.service /etc/systemd/system/mentorpi-navigation.service
sudo systemctl daemon-reload
```

The host launcher starts the service. Run a disarmed sensor and inference check first, then use the bounded motion command during a supervised trial. Goals are in metres. The example drives toward a point 0.4 m forward of the starting pose.

```bash
sudo python3 -m mentorpi_deploy 0.4
sudo python3 -m mentorpi_deploy 0.4 --drive --seconds 10
sudo python3 -m mentorpi_deploy --stop
```

A distance alone runs disarmed inference. Add `--drive` to arm, with a default 20-second timeout and a displacement limit 0.1 m beyond the requested distance. The default bundle is `isaaclab-motion-900`. Use `--policy` to select a different bundle.

Diagnostics go to `diagnostics/` and record the stop reason, inference timing, guard state and requested velocity. In observe-only mode the requested motor velocity stays zero. During policy motion, obstacle avoidance is the actor's responsibility.

## Replay a real episode

Each run saves an episode JSON under `diagnostics/` on the Pi. It includes the exact 80 policy inputs and raw actor output at every control tick, the clipped action, requested body velocity and the latest motor guard feedback. Each distinct normalized ROS scan used by the policy is stored once, with angle and range metadata and its transformed body-frame points. A null scan range marks a nonfinite return.

From the Windows workspace, copy the JSON identified by the Pi command's final output. Replace `policy-FILENAME.json` with that filename.

```powershell
scp pi@192.168.50.2:/home/pi/docker/tmp/mentorpi_isaac_deploy/diagnostics/policy-FILENAME.json deployment/diagnostics/
```

Run the following command from the deployment directory on your workstation, then open [the episode library](http://127.0.0.1:8765/). It opens the newest local recording automatically. Use the dropdown to switch episodes and **Refresh** after copying another JSON into `diagnostics/`. For a one-off file, you can also open [replay.html](replay.html) directly and choose the JSON.

```bash
python -m mentorpi_deploy.replay
```

Scrub the frame slider or press Play. Cyan dots are the LiDAR hits, and orange rays show the exact policy bins with returns under 3 m. The frame panel shows all 72 policy ranges, including bins set to 3 m. Older trace files can replay the path and existing commands.

## Calibration

Preview the diagnostic sequence on either machine. For a physical pulse test, use `--execute` inside the sourced ROS container while the deployment service is running disarmed. Check the measured directions and settled displacement against physical floor marks before changing drivetrain values. If motion or signs disagree, inspect the wheel IDs, frame transforms, and guard trace. The pulse tool does not apply a gain or alter firmware.

```bash
python3 -m mentorpi_deploy.calibrate_motion --output diagnostics/calibration.json --directions forward backward left right
python3 -m mentorpi_deploy.calibrate_motion --output diagnostics/calibration.json --directions forward backward left right --execute --check-clearance
```

The optional clearance check uses the live scan around the robot footprint. Calibration and policy execution share the same ownership lock and stop path.
