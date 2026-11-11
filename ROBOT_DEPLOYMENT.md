# LiDAR-only policy prototype

The local navigation loop is: current LiDAR scan -> robot-relative occupancy grid -> policy, together with goal, measured velocity and previous action -> body velocity -> wheel commands.

The policy uses a 100x100 occupancy grid at 0.05 m per cell. Camera depth and a saved global map are not required. RF2O laser odometry and IMU-assisted orientation track progress. Arrival is within 0.15 m of the destination; final heading is unrestricted.

After installing the deployment files on a configured Pi, run `sudo python3 run_policy_check.py --goal-relative 1 0` from its deployment directory. Coordinates are forward and left in metres. Supplying a goal starts the policy automatically after initialization. Ctrl+C stops navigation. Optional `--seconds` and `--max-displacement` add run limits; no default run limit is imposed.

This prototype leaves obstacle avoidance to the actor. External collision-zone and obstacle-distance motor overrides are disabled. Speed limits, stale-command and stale-sensor stops, and a stalled-motion detector remain. Physical goal arrival has been observed, but reliable general obstacle avoidance and long-range navigation are not established.

Device names, mount paths, serial board selection, and ROS configuration must match the target installation. Credentials and private commissioning reports are excluded from this repository. The older saved-map trial runner is separate from the direct local-goal entry point.
