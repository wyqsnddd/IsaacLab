# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import math
from .rough_env_easy_cfg import D9RoughEnvEasyCfg


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    phase_contact_reward = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "phase_contact_reward", "weight": 1.0, "num_steps": 800}
    )


@configclass
class D9RunningEnvCfg(D9RoughEnvEasyCfg):

    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Echo the default weights
        self.rewards.alive.weight = 0.8
        self.rewards.track_lin_vel_xy_exp.weight = 4
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.4
        self.rewards.feet_air_time.weight = 3
        self.rewards.no_fly.weight = 0.0
        self.rewards.phase_contact_reward.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height.weight = -5.0


        # Echo the running rewards

        self.rewards.feet_air_time_biped.weight = 0
        self.rewards.energy_efficiency.weight = 0.0

        self.rewards.alternating_air_time.weight = 0.6
        self.rewards.alternating_air_time.params["std"] = 0.4

        self.rewards.feet_slide.weight = -0.2
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -1.0

        # self.rewards.contact_no_velocity_penalty.weight = -0.005
        # self.rewards.air_time_variance_penalty.weight = -0.5
        # self.rewards.feet_swing_height.weight = -0.1

        # Running commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        self.actions.joint_pos.scale = 0.25

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "Waist_Yaw",
            ".*_Hip_.*",
            ".*_Knee_Pitch",
        ]


class D9RunningEnvCfg_PLAY(D9RunningEnvCfg):
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
