# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab_assets.external_assets.assets.pudu_d9 import PUDU_D9_15DOF_CFG # noqa F401
from isaaclab.utils import configclass
from .rough_env_cfg_15 import D9RewardsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

@configclass
class D9FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: D9RewardsCfg = D9RewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = PUDU_D9_15DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Waist_Yaw"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Waist_Yaw"]

        # terminations
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Waist_Yaw"]

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class D9FlatEnvCfg_PLAY(D9FlatEnvCfg):
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()


        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (-0, 0)
        self.commands.base_velocity.ranges.heading = (-0., 0.)

