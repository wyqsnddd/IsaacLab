from isaaclab_assets.external_assets.assets.pudu_d9 import PUDU_D9_12DOF_CFG # noqa F401

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.d9.mdp.rewards import alternating_air_time_reward, joint_deviation_l2, energy_efficiency_reward
import math

@configclass
class D9RunRewards:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.8)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.5,
        params={"command_name": "base_velocity", "std": 0.4},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.4}
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)

    alternating_air_time = RewTerm(
        func=alternating_air_time_reward,
        weight=1.6,  # 较高的权重以强调交替步态
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "period": 0.6,  # 0.6秒的步态周期
            "std": 0.4,  # 较小的标准差使奖励更敏感
            "force_threshold": 10.0,  # 接触力阈值
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

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Ankle_Roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll"),
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
        func=joint_deviation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Joint_Yaw", ".*_Hip_Joint_Roll"])},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class D9RunEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: D9RunRewards = D9RunRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.robot = PUDU_D9_12DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.events.add_base_mass.params["asset_cfg"].body_names = ["Waist_Yaw"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Waist_Yaw"]

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

        self.terminations.bad_orientation = None

class D9RunEnvCfg_PLAY(D9RunEnvCfg):
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

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (-0, 0)
        self.commands.base_velocity.ranges.heading = (-0., 0.)