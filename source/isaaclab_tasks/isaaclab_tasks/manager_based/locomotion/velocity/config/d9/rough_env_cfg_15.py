from isaaclab_assets.external_assets.assets.pudu_d9 import PUDU_D9_15DOF_CFG # noqa F401

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.d9.mdp.rewards import utils_no_fly, joint_deviation_l2, leg_arm_symmetric

import math

from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.manager_based.locomotion.velocity.config.x1.mdp.record import PreStepActionsRecorderCfg, PostStepTorqueRecorderCfg, PreStepStatesRecorderCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

@configclass
class X1RecordCfg(RecorderManagerBaseCfg):
    dataset_file_handler_class_type: type = HDF5DatasetFileHandler

    dataset_export_dir_path: str = "tmp/isaaclab/logs"
    """The directory path where the recorded datasets are exported."""

    dataset_filename: str = "d9_15dof"
    """Dataset file name without file extension."""

    dataset_export_mode: 1 # Export all episodes to a single dataset file
    """The mode to handle episode exports."""

    export_in_record_pre_reset: bool = True
    """Whether to export episodes in the record_pre_reset call."""

    record_post_step_torques = PostStepTorqueRecorderCfg()
    record_pre_step_states = PreStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()

@configclass
class D9RewardsCfg:
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
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link_Ankle_Roll"),
            "command_name": "base_velocity",
            "threshold": 0.6,
        },
    )
    
    no_fly = RewTerm(
        func = utils_no_fly,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link_Ankle_Roll"),
        },        
    )

    # base contraints
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight = -5.0,
        params={
            "target_height": 0.88,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["base_link"]
            )
        },
    )
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)

    # dof contraints
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist_Joint_Yaw", ".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist_Joint_Yaw", ".*_Hip_Joint_.*", ".*_Knee_Joint_Pitch", ".*_Ankle_Joint_.*"])})

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
                "robot", joint_names=[".*_Hip_Joint_Roll", ".*_Hip_Joint_Yaw"]
            )
        },
    )

    joint_deviation_waist = RewTerm(
        func = joint_deviation_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Waist_Joint_Yaw"]
            )
        },
    )

    joint_deviation_shoulder = RewTerm(
        func = joint_deviation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_pitch"]
            )
        },
    ) # -1

    # dof_vel_l2_shoulder = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch"])})
    # -0.005

    leg_arm_symmetric_reward = RewTerm(
        func = leg_arm_symmetric,
        weight= 0.4,
        params={
            "asset_cfg_leg": SceneEntityCfg(
                "robot", joint_names=["Left_Hip_Joint_Pitch", "Right_Hip_Joint_Pitch"]
            ),
            "asset_cfg_arm": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch", "right_shoulder_pitch"]
            )
        },
    )
    # 0.2

@configclass
class D9RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
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

        # terminations
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Waist_Yaw"]
        
@configclass
class D9RoughEnvCfg_PLAY(D9RoughEnvCfg):
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