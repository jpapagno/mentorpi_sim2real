# Project direction

This is a minimal working robotics prototype. Prioritize getting the policy to drive the car toward a user-selected goal while reacting to live obstacles.

- Make the smallest change that unblocks the working prototype. Prefer existing code and a single direct execution path.
- Keep goal selection independent of a saved map. Use LiDAR only for obstacle input: current scan → local occupancy grid; grid plus goal, measured velocity and previous action → policy → motor action. Camera depth is not part of navigation and must not gate execution.
- A supplied goal starts the policy without a confirmation prompt or a commissioning-distance restriction. Time and travel limits are optional user arguments; retain finite-number validation and essential motor protections.
- Spend effort on the actual goal → observation → policy → motor loop. Defer production hardening, extra modes, frameworks, acceptance systems, and extensive documentation unless requested.
- Do not add tests, diagnostic scripts, or readiness gates by default. Use only the focused check needed to resolve a concrete blocker; stop checking once it is resolved.
- For this small prototype, the user explicitly selected policy-only obstacle avoidance: no external obstacle-distance or collision-zone override. Preserve Ctrl+C stop, command-loss timeout, stale-sensor stops, and existing speed limits.
- Report working behavior, the command to run it, and any actual blocker briefly. Distinguish software inference from physically verified driving.
