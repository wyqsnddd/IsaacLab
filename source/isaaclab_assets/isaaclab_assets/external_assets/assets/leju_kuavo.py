# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from agent_world import ActuatorPath, AssetPath
from agent_world.actuators import DelayedPDNetActuatorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

"""Configuration for the Leju Kuavo Humanoid robot."""

LEJU_KUAVO_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/robots/leju_kuavo/kuavo_12dof.usd",
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
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "leg_l1_joint": 0.00,
            "leg_l2_joint": 0.00,
            "leg_l3_joint": -0.47,
            "leg_l4_joint": 0.86,
            "leg_l5_joint": -0.44,
            "leg_l6_joint": 0.00,
            "leg_r1_joint": 0.00,
            "leg_r2_joint": 0.00,
            "leg_r3_joint": -0.47,
            "leg_r4_joint": 0.86,
            "leg_r5_joint": -0.44,
            "leg_r6_joint": 0.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "lowerbody": DelayedPDNetActuatorCfg(
            joint_names_expr=[
                "leg_l1_joint",
                "leg_l2_joint",
                "leg_l3_joint",
                "leg_l4_joint",
                "leg_l5_joint",
                "leg_l6_joint",
                "leg_r1_joint",
                "leg_r2_joint",
                "leg_r3_joint",
                "leg_r4_joint",
                "leg_r5_joint",
                "leg_r6_joint",
            ],
            effort_limit={
                "leg_l1_joint": 180,
                "leg_l2_joint": 100,
                "leg_l3_joint": 100,
                "leg_l4_joint": 180,
                "leg_l5_joint": 36,
                "leg_l6_joint": 36,
                "leg_r1_joint": 180.0,
                "leg_r2_joint": 100.0,
                "leg_r3_joint": 100,
                "leg_r4_joint": 180,
                "leg_r5_joint": 36,
                "leg_r6_joint": 36,
            },
            velocity_limit={
                "leg_l1_joint": 14,
                "leg_l2_joint": 14,
                "leg_l3_joint": 23,
                "leg_l4_joint": 14,
                "leg_l5_joint": 10,
                "leg_l6_joint": 10,
                "leg_r1_joint": 14.0,
                "leg_r2_joint": 14.0,
                "leg_r3_joint": 23,
                "leg_r4_joint": 14,
                "leg_r5_joint": 10,
                "leg_r6_joint": 10,
            },
            stiffness={
                "leg_l1_joint": 60,
                "leg_l2_joint": 60,
                "leg_l3_joint": 60,
                "leg_l4_joint": 60,
                "leg_l5_joint": 30,
                "leg_l6_joint": 15,
                "leg_r1_joint": 60.0,
                "leg_r2_joint": 60.0,
                "leg_r3_joint": 60,
                "leg_r4_joint": 60,
                "leg_r5_joint": 30,
                "leg_r6_joint": 15,
            },
            damping={
                "leg_l1_joint": 10.0,
                "leg_l2_joint": 6.0,
                "leg_l3_joint": 12.0,
                "leg_l4_joint": 12.0,
                "leg_l5_joint": 22.0,
                "leg_l6_joint": 22.0,
                "leg_r1_joint": 10.0,
                "leg_r2_joint": 6.0,
                "leg_r3_joint": 12.0,
                "leg_r4_joint": 12.0,
                "leg_r5_joint": 22.0,
                "leg_r6_joint": 22.0,
            },
            armature=0.05,
            max_delay=20,
            joint_to_motor_position_model_path=f"{ActuatorPath}/leju_kuavo/joint_to_motor_position.pt",
            get_joint_dumping_torque_model_path=f"{ActuatorPath}/leju_kuavo/get_joint_dumping_torque.pt",
            is_ankle_joint_illegal_model_path=f"{ActuatorPath}/leju_kuavo/is_ankle_pos_legal.pt",
        ),
    },
)
