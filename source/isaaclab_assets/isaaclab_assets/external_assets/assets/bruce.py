# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from agent_world import AssetPath

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

BRUCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/robots/bruce/bruce_simplify.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.74),
    #     joint_pos={
    #         ".*_hip_pitch_joint": -0.20,
    #         ".*_knee_joint": 0.42,
    #         ".*_ankle_pitch_joint": -0.23,
    #         ".*_elbow_pitch_joint": 0.87,
    #         "left_shoulder_roll_joint": 0.16,
    #         "left_shoulder_pitch_joint": 0.35,
    #         "right_shoulder_roll_joint": -0.16,
    #         "right_shoulder_pitch_joint": 0.35,
    #         "left_one_joint": 1.0,
    #         "right_one_joint": -1.0,
    #         "left_two_joint": 0.52,
    #         "right_two_joint": -0.52,
    #     },
    #     joint_vel={".*": 0.0},
    # ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_.*",
                "knee_.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "hip_.*": 150.0,
                "knee_.*": 150.0,
            },
            damping={
                "hip_.*": 5.0,
                "knee_.*": 5.0,
            },
            armature={
                "hip_.*": 0.01,
                "knee_.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=["ankle_.*"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_.*",
                "elbow_.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                "shoulder_.*": 0.01,
                "elbow_.*": 0.01,
            },
        ),
    },
)
