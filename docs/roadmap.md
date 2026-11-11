# Robot task roadmap

MentorPi is a platform for a sequence of sim-to-real learning tasks. Each task should define an observation/action interface, trainable simulation environment and measurable hardware objective. Task 01 provides the initial perception-to-policy-to-wheel foundation; later tasks build on it.

| Task | Experiment direction | Measures to compare in simulation and on hardware |
| --- | --- | --- |
| 01 · Goal-conditioned obstacle navigation | Navigate to a position using a local obstacle grid. Initial simulations and physical demonstrations are available. | Goal error, success/contact rate, completion time and path length |
| 02 · Different-friction surfaces | Train and evaluate on wood, tile, carpet and rubber, then cross boundaries between materials. | Tracking error, wheel slip, transition recovery and goal success by surface |
| 03 · Slip and disturbance recovery | Vary traction and introduce controlled pushes or lateral disturbances. | Recovery time, drift, heading error and continued goal progress |
| 04 · Payload adaptation | Change carried mass, centre of mass and weight distribution. | Goal accuracy, response time, settling and energy use |
| 05 · Constrained-space motion | Use mecanum strafing for narrow passages, corners and parking. | Minimum clearance, contact count, final pose error and completion time |
| 06 · Moving obstacles | Introduce crossing, approaching and temporarily blocking objects. | Contact rate, clearance, yielding behavior and mission completion |
| 07 · Robust perception | Model range noise, occlusion, missing returns and observation latency. | Performance versus sensor degradation and recovery after dropout |
| 08 · Pose-accurate docking | Add target heading and approach constraints to point-goal navigation. | Position error, angular error, settling time and repeatability |
| 09 · Multi-goal missions | Follow waypoint sequences over longer runs and repeated routes. | Mission success, accumulated drift, route efficiency and localization consistency |

## Sim-to-real progression

For each planned task, first make the relevant physical effect explicit in simulation, then train on a documented range of conditions and evaluate on held-out cases. Transfer the same policy interface to hardware and compare task metrics and failure modes before widening the operating conditions.

The current model is analytic planar motion with optional mecanum response limits, actuator scaling and delay. Task 02 and Task 03 require a model of wheel-ground friction and slip, or contact-based simulation; adding a friction label to an unchanged model would not establish surface transfer. Moving obstacles, sensor degradation and final-heading control also require task/environment changes and are not current capabilities.

Future tasks are planned research directions, not completed results. See [Task 01 results](results.md) for existing evidence and [the simulation guide](simulation.md) for current tools.
