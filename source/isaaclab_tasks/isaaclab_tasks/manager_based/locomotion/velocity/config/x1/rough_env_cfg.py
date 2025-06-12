# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab_assets.external_assets.assets.x1 import X1_12DOF_CFG  # noqa F401

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.d9.mdp.rewards import (  # noqa F401
    air_time_variance_penalty,
    alternating_air_time_reward,
    biped_gait_reward,
    contact_no_velocity_penalty,
    energy_efficiency_reward,
    phase_based_contact_reward,
    utils_no_fly,
    kicking_penalty,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (  # noqa F401
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)


@configclass
class X1Rewards:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.4,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.6, params={"command_name": "base_velocity", "std": 0.4}
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)

    # Penalize kicking stairs
    kicking_penalty = RewTerm(
        func=kicking_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "friction_coefficient": 0.5,
            "std": 0.3,
        },
    )

    # Rewards for running
    feet_air_time_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "threshold": 0.1,
        },
    )

    alternating_air_time = RewTerm(
        func=alternating_air_time_reward,
        weight=0.0,  # 较高的权重以强调交替步态
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "period": 0.6,  # 0.6秒的步态周期
            "std": 0.3,  # 较小的标准差使奖励更敏感
            "force_threshold": 10.0,  # 接触力阈值
        },
    )
    # 能量效率奖励
    energy_efficiency = RewTerm(
        func=energy_efficiency_reward,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "threshold": 0.6,
        },
    )

    no_fly = RewTerm(
        func=utils_no_fly,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link_Ankle_Roll"),
        },
    )

    # Add phase-based contact reward
    phase_contact_reward = RewTerm(
        func=phase_based_contact_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "period": 0.8,
            "offset": 0.5,
            "std": 0.5,
            "force_threshold": 1.0,
        },
    )
    # Add contact no velocity penalty
    contact_no_velocity_penalty = RewTerm(
        func=contact_no_velocity_penalty,
        weight=0.0,  # Negative weight since this is a penalty
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll"),
            "force_threshold": 1.0,
        },
    )

    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll"),
            "target_height": 0.12,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll"),
        },
    )
    # Add air time variance penalty
    air_time_variance_penalty = RewTerm(
        func=air_time_variance_penalty,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "std": 0.1,
        },
    )

    # Add gait reward for bipedal walking
    gait_reward = RewTerm(
        func=biped_gait_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll"),
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.1,
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "target_height": 0.88,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"]
            )
        },
    )

    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"]
            )
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"]
            )
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"]
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Joint_Yaw", ".*_Hip_Joint_Roll"])},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0)


@configclass
class X1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: X1Rewards = X1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = X1_12DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = ["waist-j"]

        # self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist-j"]

        # Echo the default weights
        self.rewards.alive.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.4
        self.rewards.track_ang_vel_z_exp.weight = 1.6
        self.rewards.feet_air_time.weight = 3.0
        self.rewards.no_fly.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height.weight = -5.0
        self.rewards.phase_contact_reward.weight = 1.0
        self.rewards.feet_slide.weight = -0.0
        self.rewards.contact_no_velocity_penalty.weight = -0.0
        self.rewards.air_time_variance_penalty.weight = -0.0
        self.rewards.feet_swing_height.weight = -0.0
        self.rewards.ang_vel_xy_l2.weight = -1.0

        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.kicking_penalty.weight = -0.5

        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }

        # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

        # terminations
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "waist-j",
            "leg-j1-.*",
            "leg-j2-.*",
            "leg-j3-.*",
            "leg-j4-.*",
        ]

        # Add bad orientation termination
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={
                # "limit_angle": 0.436,  # Limit angle in radians (approximately 25 degrees)
                "limit_angle": 1.0,  # Limit angle in radians (approximately 57 degrees)
                "asset_cfg": SceneEntityCfg("robot", body_names=["waist-j"]),
            },
            time_out=False,
        )

        self.terminations.bad_orientation = None


@configclass
class X1RoughEnvCfg_PLAY(X1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
