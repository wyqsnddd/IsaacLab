# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from agent_world import AssetPath

from isaaclab_assets import G1_MINIMAL_CFG  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

"""Configuration for the Unitree G1 Humanoid robot."""

UNITREE_G1_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/robots/unitree_g1/g1_12dof.usd",
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
            fix_root_link=False,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

UNITREE_G1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/robots/unitree_g1/g1_29dof_lock_wrist_rev_1_0.usd",
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
            fix_root_link=False,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_ankle_roll_joint": 0.0,
            "waist_.*_joint": 0.0,
            ".*_elbow_joint": 1.4,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waists": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_.*_joint",
                ".*_knee_joint",
            ],
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*joint",
                ".*_elbow_joint",
            ],
            stiffness={
                ".*_shoulder_.*joint": 80.0,
                ".*_elbow_joint": 60.0,
            },
            damping={
                ".*_shoulder_.*joint": 3.0,
                ".*_elbow_joint": 2.0,
            },
            armature=0.01,
        ),
    },
)

UNITREE_G1_27DOF_CFG = UNITREE_G1_23DOF_CFG.copy()
UNITREE_G1_27DOF_CFG.spawn.usd_path = f"{AssetPath}/usd_files/robots/unitree_g1/g1_29dof_lock_waist_rev_1_0.usd"
UNITREE_G1_27DOF_CFG.init_state.joint_pos[".*_wrist_.*_joint"] = 0.0
UNITREE_G1_27DOF_CFG.actuators["arms"].joint_names_expr.append(".*_wrist_.*_joint")
UNITREE_G1_27DOF_CFG.actuators["arms"].stiffness[".*_wrist_.*_joint"] = 40.0
UNITREE_G1_27DOF_CFG.actuators["arms"].damping[".*_wrist_.*_joint"] = 2.0

UNITREE_G1_29DOF_CFG = UNITREE_G1_27DOF_CFG.copy()
UNITREE_G1_29DOF_CFG.spawn.usd_path = f"{AssetPath}/usd_files/robots/unitree_g1/g1_29dof_rev_1_0.usd"

UNITREE_G1_27DOF_ROBOTIQ_CFG = UNITREE_G1_27DOF_CFG.copy()
UNITREE_G1_27DOF_ROBOTIQ_CFG.spawn.usd_path = f"{AssetPath}/usd_files/robots/unitree_g1_robotiq/g1_27dof_v2_flatten.usd"
UNITREE_G1_27DOF_ROBOTIQ_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_finger_joint", ".*_knuckle_joint"],
    stiffness=1000,
    damping=10,
)
