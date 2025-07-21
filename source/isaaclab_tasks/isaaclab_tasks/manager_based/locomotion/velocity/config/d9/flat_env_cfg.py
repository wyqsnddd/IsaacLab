# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import D9RoughEnvCfg


@configclass
class D9FlatEnvCfg(D9RoughEnvCfg):
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
        self.rewards.gait_reward.weight = 0.5
        self.rewards.phase_contact_reward.weight = 1
        self.rewards.contact_no_velocity_penalty.weight = -0.005
        self.rewards.air_time_variance_penalty.weight = -0.5
        self.rewards.base_height.weight = -2.0
        self.rewards.feet_swing_height.weight = -0.1
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1e-05
        self.rewards.dof_acc_l2.weight = -2.5e-07
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.feet_air_time.params["threshold"] = 0.6
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.termination_penalty.weight = -200
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_hip_2.weight = 0.0
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_arms_2.weight = 0.0
        self.rewards.joint_deviation_torso.weight = -0.1
        self.rewards.joint_deviation_torso_2.weight = 0.0


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
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0, 0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
