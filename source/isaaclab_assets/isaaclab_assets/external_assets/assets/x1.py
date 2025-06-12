# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.external_assets import ExternalAssetLoader

loader = ExternalAssetLoader()
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

"""Configuration for the X1 Humanoid robot."""

X1_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=loader.get_robot_usd_path("x1", "x1_12dof.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            "leg-j1_r": 0.3,
            "leg-j1_l": -0.3,
            "leg-j2_.*": 0.0,
            "leg-j3_.*": 0.0,
            "leg-j4_r": 0.7,
            "leg-j4_l": -0.7,
            "leg-j5_r": -0.4,
            "leg-j5_l": 0.4,
            "leg-j6_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg-j.*",
            ],
            stiffness={
                "leg-j1_.*": 200.0,
                "leg-j3.*": 150.0,
                "leg-j2.*": 150.0,
                "leg-j4.*": 200.0,
            },
            damping={
                "leg-j1_.*": 5,
                "leg-j3_.*": 5,
                "leg-j2_.*": 5,
                "leg-j4_.*": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["leg-j5.*", "leg-j6.*"],
            stiffness={
                "leg-j5.*": 20.0,
                "leg-j6.*": 20.0,
            },
            damping={
                "leg-j5.*": 2.0,
                "leg-j6.*": 2.0,
            },
            armature=0.01,
        ),
    },
)
