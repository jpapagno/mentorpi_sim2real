# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""

from isaaclab_isaacsim.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

MENTORPI_RECT_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 0.2, 0.15),
        rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)],
        collision_props=[sim_utils.UsdPhysicsCollisionCfg()],
        mass_props=[sim_utils.MassCfg(mass=1.0)],
        activate_contact_sensors=True
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.04)),
)

MENTORPI_LIDAR_CFG = MultiMeshRayCasterCfg(
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
    ray_alignment="yaw",
    pattern_cfg=LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=5.0,
    ),
    max_distance=3.0,
    update_period=0.1,
)

"""Configuration for a simple Cartpole robot."""
