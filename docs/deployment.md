# Robot deployment

The runtime targets the Hiwonder MentorPi M1 with a Raspberry Pi, ROS2 Humble container, LiDAR and RRC motor controller. It implements the same goal-conditioned policy interface used in simulation.

## Runtime pipeline

`scan_normalizer` → LiDAR occupancy + measured state → `policy` → body velocity → `hardware_bridge` → wheel channels.

- `navigation_io` transforms the current scan into the body frame, builds the occupancy grid and supplies goal, measured velocity and previous action.
- RF2O estimates motion from successive scans. `measured_odometry` derives body velocity; the EKF combines measured pose with IMU rotation information.
- `policy` runs the exported TorchScript actor. `ownership` coordinates the single active policy runner and idle disarmed state.
- `hardware_bridge` owns serial output and applies the shared mecanum conversion, speed/acceleration limits and command/sensor expiry.

The default loop does not require a saved global map or camera depth. The exported actor belongs in `models/policy.ts`, alongside the export/selection metadata. Credentials, weights and generated run reports are excluded from source control.

## Goal commands

Run from the installed repository root on the Pi:

```bash
# A goal one metre ahead of the starting pose.
sudo python3 -m mentorpi_nav.robot --goal-relative 1 0

# A goal ahead and to the left, with optional run limits.
sudo python3 -m mentorpi_nav.robot --goal-relative 0.6 0.2 --seconds 20 --max-displacement 0.8

# A fixed point in the current odometry frame.
sudo python3 -m mentorpi_nav.robot --goal-odom 1.2 -0.4

# Stop the navigation service and motor output.
sudo python3 -m mentorpi_nav.robot --stop
```

Coordinates are metres: relative values are forward and left. The relative target is anchored once in odometry coordinates; it does not move forward with the robot. Odometry resets on stack restart and can drift. Arrival means within 0.15 m of the target, without a final heading constraint.

A supplied goal starts automatically after model and sensor initialization. Ctrl+C stops navigation. Time and displacement limits are optional. The runtime retains command expiry, stale-sensor stops, actuator limits and a no-measured-motion detector. Obstacle avoidance is currently selected by the actor, without an external collision-zone or obstacle-distance override. Accepting a long-distance goal is not evidence of reliable long-range navigation.

## Configuration and installation

`deployment/config/` contains the active LiDAR, RF2O, EKF and drivetrain settings. Verify board selection in `robot/rrc_serial.py`, frame names, wheel geometry, speed limits and container paths for the target installation. Motor order is front-left, rear-left, rear-right, front-right, with forward signs `(-1,-1,+1,+1)`.

Copy the `mentorpi_nav/` package and `deployment/` directory into the configured host workspace, preserving existing `models/` files. That root must remain visible through the configured container mount. Then install the supplied service from the checkout:

```bash
sudo systemctl stop mentorpi-navigation.service
sudo install -m 644 deployment/mentorpi-navigation.service /etc/systemd/system/mentorpi-navigation.service
sudo systemctl daemon-reload
sudo systemctl start mentorpi-navigation.service
```

The service starts disarmed and launches `deployment/navigation.launch.py`; the implementation is in `mentorpi_nav.robot` modules. Repository updates alone do not change an already installed robot. Keep sensor transforms and measured odometry available before policy execution.

SSH transport is `mentorpi_nav.tools.ssh`; the Windows wrapper `scripts/robot-ssh.ps1` loads ignored local connection settings and Windows-protected credentials. Simulation, robot inference and SSH dependencies are separate environments.

See [results](results.md) for measured evidence and [Trial 1 in the README](../README.md#2-real-robot-around-an-obstacle) for the first task's physical demonstration.
