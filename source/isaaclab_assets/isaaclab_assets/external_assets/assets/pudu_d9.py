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


PUDU_D9_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=loader.get_robot_usd_path("pudu_d9", "d9_12dof.usd"),
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
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            ".*_Hip_Joint_Pitch": 0.3,
            ".*_Hip_Joint_Roll": 0.0,
            ".*_Hip_Joint_Yaw": 0.0,
            ".*_Knee_Joint_Pitch": -0.7,
            ".*_Ankle_Joint_Pitch": 0.4,
            ".*_Ankle_Joint_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Joint_Pitch",
                ".*_Hip_Joint_Roll",
                ".*_Hip_Joint_Yaw",
                ".*_Knee_Joint_Pitch",
            ],
            stiffness={
                ".*_Hip_Joint_Pitch": 200.0,
                ".*_Hip_Joint_Roll": 150.0,
                ".*_Hip_Joint_Yaw": 150.0,
                ".*_Knee_Joint_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Joint_Pitch": 5.0,
                ".*_Hip_Joint_Roll": 5.0,
                ".*_Hip_Joint_Yaw": 5.0,
                ".*_Knee_Joint_Pitch": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Joint_Pitch", ".*_Ankle_Joint_Roll"],
            stiffness={
                ".*_Ankle_Joint_Pitch": 20.0,
                ".*_Ankle_Joint_Roll": 20.0,
            },
            damping={
                ".*_Ankle_Joint_Pitch": 2.0,
                ".*_Ankle_Joint_Roll": 2.0,
            },
            armature=0.01,
        ),
    },
)

PUDU_D9_15DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=loader.get_robot_usd_path("pudu_d9", "d9_15dof.usd"),
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
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            ".*_Hip_Joint_Pitch": 0.3,
            ".*_Hip_Joint_Roll": 0.0,
            ".*_Hip_Joint_Yaw": 0.0,
            ".*_Knee_Joint_Pitch": -0.7,
            ".*_Ankle_Joint_Pitch": 0.4,
            ".*_Ankle_Joint_Roll": 0.0,
            ".*_shoulder_pitch": 0.0,
            "Waist_Joint_Yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Joint_Pitch",
                ".*_Hip_Joint_Roll",
                ".*_Hip_Joint_Yaw",
                ".*_Knee_Joint_Pitch",
            ],
            stiffness={
                ".*_Hip_Joint_Pitch": 200.0,
                ".*_Hip_Joint_Roll": 150.0,
                ".*_Hip_Joint_Yaw": 150.0,
                ".*_Knee_Joint_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Joint_Pitch": 5.0,
                ".*_Hip_Joint_Roll": 5.0,
                ".*_Hip_Joint_Yaw": 5.0,
                ".*_Knee_Joint_Pitch": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Joint_Pitch", ".*_Ankle_Joint_Roll"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "extras": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", "Waist_Joint_Yaw"],
            stiffness={
                ".*_shoulder_pitch": 20.0,
                "Waist_Joint_Yaw": 150,
            },
            damping={
                ".*_shoulder_pitch": 2.0,
                "Waist_Joint_Yaw": 5.0,
            },
            armature=0.01,
        ),
    },
)

PUDU_D9_17DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=loader.get_robot_usd_path("pudu_d9", "d9_17dof.usd"),
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
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            ".*_Hip_Joint_Pitch": 0.3,
            ".*_Hip_Joint_Roll": 0.0,
            ".*_Hip_Joint_Yaw": 0.0,
            ".*_Knee_Joint_Pitch": -0.7,
            ".*_Ankle_Joint_Pitch": 0.4,
            ".*_Ankle_Joint_Roll": 0.0,
            ".*_shoulder_pitch": 0.0,
            "right_shoulder_roll": -0.1,
            "left_shoulder_roll": 0.1,
            "Waist_Joint_Yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Joint_Pitch",
                ".*_Hip_Joint_Roll",
                ".*_Hip_Joint_Yaw",
                ".*_Knee_Joint_Pitch",
            ],
            stiffness={
                ".*_Hip_Joint_Pitch": 200.0,
                ".*_Hip_Joint_Roll": 150.0,
                ".*_Hip_Joint_Yaw": 150.0,
                ".*_Knee_Joint_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Joint_Pitch": 5.0,
                ".*_Hip_Joint_Roll": 5.0,
                ".*_Hip_Joint_Yaw": 5.0,
                ".*_Knee_Joint_Pitch": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Joint_Pitch", ".*_Ankle_Joint_Roll"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "extras": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", "Waist_Joint_Yaw"],
            stiffness={
                ".*_shoulder_pitch": 20.0,
                ".*_shoulder_roll": 20.0,
                "Waist_Joint_Yaw": 150,
            },
            damping={
                ".*_shoulder_pitch": 2.0,
                ".*_shoulder_roll": 2.0,
                "Waist_Joint_Yaw": 5.0,
            },
            armature=0.01,
        ),
    },
)
