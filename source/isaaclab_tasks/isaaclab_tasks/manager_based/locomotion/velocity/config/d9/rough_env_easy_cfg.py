# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
# TODO: import PUDU_D9
# from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab_assets.external_assets.assets.pudu_d9 import PUDU_D9_12DOF_UPDATED_ACTUATOR_CFG  # noqa F401

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.d9.mdp.rewards import (  # noqa F401
    air_time_variance_penalty,
    biped_gait_reward,
    contact_no_velocity_penalty,
    phase_based_contact_reward,
    utils_no_fly,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (  # noqa F401
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)


@configclass
class D9Rewards:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
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
        weight=0.25,
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
            "std": 0.1,
            "force_threshold": 1.0,
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


@configclass
class D9RoughEnvEasyCfg(LocomotionVelocityRoughEnvCfg):
    rewards: D9Rewards = D9Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = PUDU_D9_12DOF_UPDATED_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Waist_Yaw"]

        # self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Waist_Yaw"]

        # Echo the default weights
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.feet_air_time.weight = 3.0
        self.rewards.no_fly.weight = 0.25
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height.weight = -5.0
        self.phase_contact_reward.weight = 1.0
        self.rewards.feet_slide.weight = -0.0
        self.rewards.contact_no_velocity_penalty.weight = -0.0
        self.rewards.air_time_variance_penalty.weight = -0.0
        self.rewards.feet_swing_height.weight = -0.0

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
            "Waist_Yaw",
            ".*_Hip_.*",
            ".*_Knee_Pitch",
        ]

        # Add bad orientation termination
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={
                # "limit_angle": 0.436,  # Limit angle in radians (approximately 25 degrees)
                "limit_angle": 1.0,  # Limit angle in radians (approximately 57 degrees)
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "Waist_Yaw"]),
            },
            time_out=False,
        )


@configclass
class D9RoughEnvEasyCfg_PLAY(D9RoughEnvEasyCfg):
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
