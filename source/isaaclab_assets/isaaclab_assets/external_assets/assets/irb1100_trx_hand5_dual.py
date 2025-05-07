# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from agent_world import AssetPath

from isaaclab_assets import G1_MINIMAL_CFG  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

"""Configuration for the irb1100_trx_hand5_dual robot."""

IRB1100_TRX_HAND5_DUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/robots/irb1100_trx_hand5_dual/irb1100_trx_hand5_dual.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        # fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_(L|R)\\d",
                "J_F0_(CMR|CMP|MPP|IP)_(L|R)",
                "J_F(1|2)_(MPR|MPP|PIP|DIP)_(L|R)",
                "J_F(3|4)_(MPP|PIP|DIP)_(L|R)",
            ],
            stiffness={
                "J_(L|R)(1|2)": 50,
                "J_(L|R)(3|4)": 50,
                "J_(L|R)(5|6)": 30,
                "J_F0_(CMR|CMP)_(L|R)": 6.0,
                "J_F0_(MPP|IP)_(L|R)": 6.0,
                "J_F(1|2)_(MPR|MPP)_(L|R)": 6.0,
                "J_F(1|2)_(PIP|DIP)_(L|R)": 6.0,
                "J_F(3|4)_MPP_(L|R)": 2.0,
                "J_F(3|4)_(PIP|DIP)_(L|R)": 2.0,
            },
            damping={
                "J_(L|R)(1|2)": 3,
                "J_(L|R)(3|4)": 1,
                "J_(L|R)(5|6)": 0.5,
                "J_F0_(CMR|CMP)_(L|R)": 0.2,
                "J_F0_(MPP|IP)_(L|R)": 0.2,
                "J_F(1|2)_(MPR|MPP)_(L|R)": 0.2,
                "J_F(1|2)_(PIP|DIP)_(L|R)": 0.2,
                "J_F(3|4)_MPP_(L|R)": 0.05,
                "J_F(3|4)_(PIP|DIP)_(L|R)": 0.05,
            },
        ),
    },
)

IRB1100_TRX_HAND5_DUAL_TACTILES_CFG = IRB1100_TRX_HAND5_DUAL_CFG.copy()
IRB1100_TRX_HAND5_DUAL_TACTILES_CFG.spawn.usd_path = (
    f"{AssetPath}/usd_files/robots/irb1100_trx_hand5_dual/irb1100_trx_hand5_dual_with_tactiles.usd"
)
