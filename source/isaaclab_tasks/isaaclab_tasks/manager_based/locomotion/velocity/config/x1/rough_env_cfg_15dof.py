from isaaclab_assets.external_assets.assets.x1 import X1_15DOF_CFG

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.x1.mdp.rewards import utils_no_fly, joint_deviation_l2, leg_arm_symmetric
import math


@configclass
class X1Rewards15DOFCfg:
    """Reward terms for the MDP."""
    # main reward
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Ankle_Roll_.*"),
            "command_name": "base_velocity",
            "threshold": 0.6,
        },
    )
    
    no_fly = RewTerm(
        func = utils_no_fly,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Ankle_Roll_.*"),
        },        
    )

    # base contraints
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight = -5.0,
        params={
            "target_height": 1.08,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["Base_Link"]
            )
        },
    )
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)

    # dof contraints
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    
    joint_deviation_hip = RewTerm(
        func = joint_deviation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Joint_Hip_Roll_.*", "Joint_Hip_Yaw_.*"]
            )
        },
    )

    joint_deviation_extra = RewTerm(
        func = joint_deviation_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Joint_Shoulder_Pitch_.*"]
            )
        },
    )

    leg_arm_symmetric_reward = RewTerm(
        func = leg_arm_symmetric,
        weight= -0.1,
        params={
            "asset_cfg_leg": SceneEntityCfg(
                "robot", joint_names=["Joint_Hip_Pitch_Left", "Joint_Hip_Pitch_Right"]
            ),
            "asset_cfg_arm": SceneEntityCfg(
                "robot", joint_names=["Joint_Shoulder_Pitch_Left", "Joint_Hip_Pitch_Right"]
            )
        },
    )


@configclass
class X1RoughEnv15DofCfg(LocomotionVelocityRoughEnvCfg):
    rewards: X1Rewards15DOFCfg = X1Rewards15DOFCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = X1_15DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Base_Link"

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Base_Link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Base_Link"]

        # terminations
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Base_Link"]
        
@configclass
class X1RoughEnv15Dof_PLAY(X1RoughEnv15DofCfg):
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

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None