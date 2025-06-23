# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.external_assets import ExternalAssetLoader

loader = ExternalAssetLoader()
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

"""Configuration for the Pudu D9 Humanoid robot."""

X1_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=loader.get_robot_usd_path("X1", "X1_12dof.usd"),
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.08),
        joint_pos={
            "Joint_Hip_Pitch_.*": -0.2,
            "Joint_Hip_Roll_.*": 0.0,
            "Joint_Hip_Yaw_.*": 0.0,
            "Joint_Knee_Pitch_.*": 0.6,
            "Joint_Ankle_Pitch_.*": -0.4,
            "Joint_Ankle_Roll_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "Joint_Hip_Pitch_.*",
                "Joint_Hip_Roll_.*",
                "Joint_Hip_Yaw_.*",
                "Joint_Knee_Pitch_.*",
            ],
            stiffness={
                "Joint_Hip_Pitch_.*": 200.0,
                "Joint_Hip_Roll_.*": 150.0,
                "Joint_Hip_Yaw_.*": 150.0,
                "Joint_Knee_Pitch_.*": 200.0,
            },
            damping={
                "Joint_Hip_Pitch_.*": 5.0,
                "Joint_Hip_Roll_.*": 5.0,
                "Joint_Hip_Yaw_.*": 5.0,
                "Joint_Knee_Pitch_.*": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["Joint_Ankle_Pitch_.*", "Joint_Ankle_Roll_.*"],
            stiffness={
                "Joint_Ankle_Pitch_.*": 20.0,
                "Joint_Ankle_Roll_.*": 20.0,
            },
            damping={
                "Joint_Ankle_Pitch_.*": 2.0,
                "Joint_Ankle_Roll_.*": 2.0,
            },
            armature=0.01,
        ),
    },
)

X1_15DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=loader.get_robot_usd_path("X1", "X1_15dof.usd"),
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.08),
        joint_pos={
            "Joint_Hip_Pitch_.*": -0.2,
            "Joint_Hip_Roll_.*": 0.0,
            "Joint_Hip_Yaw_.*": 0.0,
            "Joint_Knee_Pitch_.*": 0.6,
            "Joint_Ankle_Pitch_.*": -0.4,
            "Joint_Ankle_Roll_.*": 0.0,
            "Joint_Shoulder_Pitch_.*": 0.0,
            "Joint_Waist_Yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "Joint_Hip_Pitch_.*",
                "Joint_Hip_Roll_.*",
                "Joint_Hip_Yaw_.*",
                "Joint_Knee_Pitch_.*",
            ],
            stiffness={
                "Joint_Hip_Pitch_.*": 200.0,
                "Joint_Hip_Roll_.*": 150.0,
                "Joint_Hip_Yaw_.*": 150.0,
                "Joint_Knee_Pitch_.*": 200.0,
            },
            damping={
                "Joint_Hip_Pitch_.*": 5.0,
                "Joint_Hip_Roll_.*": 5.0,
                "Joint_Hip_Yaw_.*": 5.0,
                "Joint_Knee_Pitch_.*": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["Joint_Ankle_Pitch_.*", "Joint_Ankle_Roll_.*"],
            stiffness={
                "Joint_Ankle_Pitch_.*": 20.0,
                "Joint_Ankle_Roll_.*": 20.0,
            },
            damping={
                "Joint_Ankle_Pitch_.*": 2.0,
                "Joint_Ankle_Roll_.*": 2.0,
            },
            armature=0.01,
        ),
        "extras": ImplicitActuatorCfg(
            joint_names_expr=["Joint_Shoulder_Pitch_.*", "Joint_Waist_Yaw"],
            stiffness={
                "Joint_Shoulder_Pitch_.*": 20.0,
                "Joint_Waist_Yaw": 150.0,
            },
            damping={
                "Joint_Shoulder_Pitch_.*": 2.0,
                "Joint_Waist_Yaw": 5.0,
            },
            armature=0.01,
        ),
    },
)