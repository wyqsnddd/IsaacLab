# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.external_assets.assets.pudu_d9 import PUDU_D9_15DOF_CFG  # noqa F401

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (  # noqa F401
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.d9.mdp.rewards import (  # noqa F401
    air_time_variance_penalty,
    alternating_air_time_reward,
    biped_gait_reward,
    contact_no_velocity_penalty,
    energy_efficiency_reward,
    phase_based_contact_reward,
    utils_no_fly,
    kicking_penalty,
    leg_arm_symmetric,
)

import math

from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.manager_based.locomotion.velocity.config.x1.mdp.record import (
    PreStepActionsRecorderCfg,
    PostStepTorqueRecorderCfg,
    PreStepStatesRecorderCfg,
)
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg


@configclass
class X1RecordCfg(RecorderManagerBaseCfg):
    dataset_file_handler_class_type: type = HDF5DatasetFileHandler

    dataset_export_dir_path: str = "tmp/isaaclab/logs"
    """The directory path where the recorded datasets are exported."""

    dataset_filename: str = "d9_15dof"
    """Dataset file name without file extension."""

    dataset_export_mode: 1  # Export all episodes to a single dataset file
    """The mode to handle episode exports."""

    export_in_record_pre_reset: bool = True
    """Whether to export episodes in the record_pre_reset call."""

    record_post_step_torques = PostStepTorqueRecorderCfg()
    record_pre_step_states = PreStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()


@configclass
class D9RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    # main reward
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.4, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.6, params={"command_name": "base_velocity", "std": math.sqrt(0.4)}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link_Ankle_Roll"),
            "command_name": "base_velocity",
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
    # base constraints
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    flat_orientation_l1 = RewTerm(func=mdp.flat_orientation_l1, weight=-1.0)

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.88, "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    )
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)

    # dof constraints
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Waist_Joint_Yaw", ".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"]
            )
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Waist_Joint_Yaw", ".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"]
            )
        },
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Joint_Roll", ".*_Hip_Joint_Yaw"])},
    )

    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist_Joint_Yaw"])},
    )

    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch"])},
    )  # -1

    # dof_vel_l2_shoulder = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch"])})
    # -0.005

    leg_arm_symmetric_reward = RewTerm(
        func=leg_arm_symmetric,
        weight=0.4,
        params={
            "asset_cfg_leg": SceneEntityCfg("robot", joint_names=["Left_Hip_Joint_Pitch", "Right_Hip_Joint_Pitch"]),
            "asset_cfg_arm": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch", "right_shoulder_pitch"]),
        },
    )
    # 0.2


@configclass
class D9RoughEnvCfg15(LocomotionVelocityRoughEnvCfg):
    rewards: D9RewardsCfg = D9RewardsCfg()
    # recorders: object = X1RecordCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = PUDU_D9_15DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Waist_Yaw"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Waist_Yaw"]

        # 单独修改楼梯宽度
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_width = 0.5
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_width = 0.5
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.65

        # Echo the default weights
        self.rewards.alive.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.4
        self.rewards.track_ang_vel_z_exp.weight = 1.6
        self.rewards.feet_air_time.weight = 3.0
        self.rewards.no_fly.weight = 0.75
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.flat_orientation_l1.weight = -2.0
        self.rewards.base_height.weight = -5.0
        self.rewards.phase_contact_reward.weight = 1.0
        # self.rewards.ang_vel_xy_l2.weight = -1.0

        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7

        self.rewards.kicking_penalty.weight = -0.5

        # terminations
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "Waist_Yaw",
            # ".*_Hip_.*",
            # ".*_Knee_Pitch",
            # "base_link",
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
class D9RoughEnvCfg15_PLAY(D9RoughEnvCfg15):
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

        # 单独修改楼梯宽度
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_width = 0.5
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_width = 0.5
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.65

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
