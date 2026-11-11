# LiDAR-only policy prototype

The local navigation loop is: current LiDAR scan -> robot-relative occupancy grid -> policy, together with goal, measured velocity and previous action -> body velocity -> wheel commands.

The policy uses a 100x100 occupancy grid at 0.05 m per cell. Camera depth and a saved global map are not required. RF2O laser odometry and IMU-assisted orientation track progress. Arrival is within 0.15 m of the destination; final heading is unrestricted.

After installing the deployment files on a configured Pi, run `sudo python3 -m mentorpi_nav.robot --goal-relative 1 0` from the repository root. Coordinates are forward and left in metres. Supplying a goal starts the policy automatically after initialization. Ctrl+C stops navigation. Optional `--seconds` and `--max-displacement` add run limits; no default run limit is imposed.

This prototype leaves obstacle avoidance to the actor. External collision-zone and obstacle-distance motor overrides are disabled. Speed limits, stale-command and stale-sensor stops, and a stalled-motion detector remain. Physical goal arrival has been observed, but reliable general obstacle avoidance and long-range navigation are not established.

Device names, mount paths, serial board selection, and ROS configuration must match the target installation. Credentials and private commissioning reports are excluded from this repository. Obsolete saved-map trials, camera navigation, and one-off deployment probes are removed.

## Installing the reorganized checkout

The existing Pi installation is not automatically modified by a repository cleanup. Copy the `mentorpi_nav/` package and `deployment/` directory into the configured host workspace, keeping its existing `models/` files. The same root must remain visible at the configured container mount. From that root on the Pi:

```bash
sudo systemctl stop mentorpi-navigation.service
sudo install -m 644 deployment/mentorpi-navigation.service /etc/systemd/system/mentorpi-navigation.service
sudo systemctl daemon-reload
sudo systemctl start mentorpi-navigation.service
sudo python3 -m mentorpi_nav.robot --goal-relative 1 0
```

The service launches `deployment/navigation.launch.py`; all executable behavior lives in `mentorpi_nav.robot` modules. Stop with Ctrl+C or `sudo python3 -m mentorpi_nav.robot --stop`. Existing speed, freshness, stall and motor-stop behavior is retained.
