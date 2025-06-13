# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (  # noqa F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


def d9_leg_symmetry_augmentation(env, obs, actions, obs_type="policy"):
    """D9机器人腿部对称性数据增强函数"""
    # 添加详细的调试信息
    print(f"\n=== Symmetry Augmentation Debug Info ===")  # noaq F541
    print(f"obs_type: {obs_type}")
    print(f"obs shape: {obs.shape if obs is not None else 'None'}")
    print(f"actions shape: {actions.shape if actions is not None else 'None'}")

    # 根据obs_type处理不同的情况
    if obs_type == "critic":
        # critic只需要处理obs，不需要处理actions
        if obs is None:
            return None, None
        # 对于critic，我们仍然需要返回镜像的obs以计算对称性损失
        mirrored_obs = obs.clone()
        # 定义关节的对称变换系数
        joint_multipliers = {
            # 左腿关节
            0: -1.0,  # Left_Hip_Joint_Roll
            1: -1.0,  # Left_Hip_Joint_Yaw
            2: 1.0,  # Left_Hip_Joint_Pitch
            3: 1.0,  # Left_Knee_Joint_Pitch
            4: 1.0,  # Left_Ankle_Joint_Pitch
            5: -1.0,  # Left_Ankle_Joint_Roll
            # 右腿关节
            6: -1.0,  # Right_Hip_Joint_Roll
            7: -1.0,  # Right_Hip_Joint_Yaw
            8: 1.0,  # Right_Hip_Joint_Pitch
            9: 1.0,  # Right_Knee_Joint_Pitch
            10: 1.0,  # Right_Ankle_Joint_Pitch
            11: -1.0,  # Right_Ankle_Joint_Roll
        }

        # 对每个关节应用对称变换
        for i in range(12):  # D9有12个关节
            if i in joint_multipliers:
                # 处理观察值中的关节位置和速度
                mirrored_obs[:, i] = obs[:, i] * joint_multipliers[i]  # 关节位置
                mirrored_obs[:, i + 12] = obs[:, i + 12] * joint_multipliers[i]  # 关节速度

        # 检查输出是否有效
        if torch.isnan(mirrored_obs).any() or torch.isinf(mirrored_obs).any():
            print("Warning: Output obs contain NaN or Inf values")
            return None, None

        return mirrored_obs, None

    # 对于policy类型，需要处理actions，obs可以为None
    if actions is None:
        return None, None

    # 检查输入类型
    if not isinstance(actions, torch.Tensor):
        return None, None

    # 打印输入统计信息
    try:
        if obs is not None:
            print(
                f"Input obs - mean: {obs.mean():.4f}, std: {obs.std():.4f}, min: {obs.min():.4f}, max: {obs.max():.4f}"
            )
        print(
            f"Input actions - mean: {actions.mean():.4f}, std: {actions.std():.4f}, min: {actions.min():.4f}, max: {actions.max():.4f}"
        )
    except Exception as e:
        print(f"Warning: Error printing statistics: {e}")

    # 添加数值检查
    if torch.isnan(actions).any() or torch.isinf(actions).any():
        print("Warning: Input actions contain NaN or Inf values")
        return None, None

    # 定义关节的对称变换系数
    joint_multipliers = {
        # 左腿关节
        0: -1.0,  # Left_Hip_Joint_Roll
        1: -1.0,  # Left_Hip_Joint_Yaw
        2: 1.0,  # Left_Hip_Joint_Pitch
        3: 1.0,  # Left_Knee_Joint_Pitch
        4: 1.0,  # Left_Ankle_Joint_Pitch
        5: -1.0,  # Left_Ankle_Joint_Roll
        # 右腿关节
        6: -1.0,  # Right_Hip_Joint_Roll
        7: -1.0,  # Right_Hip_Joint_Yaw
        8: 1.0,  # Right_Hip_Joint_Pitch
        9: 1.0,  # Right_Knee_Joint_Pitch
        10: 1.0,  # Right_Ankle_Joint_Pitch
        11: -1.0,  # Right_Ankle_Joint_Roll
    }

    # 创建镜像观察值和动作
    mirrored_obs = obs.clone() if obs is not None else None
    mirrored_actions = actions.clone()

    # 对每个关节应用对称变换
    for i in range(12):  # D9有12个关节
        if i in joint_multipliers:
            # 处理动作
            mirrored_actions[:, i] = actions[:, i] * joint_multipliers[i]

            # 如果obs不为None，处理观察值中的关节位置和速度
            if mirrored_obs is not None:
                mirrored_obs[:, i] = obs[:, i] * joint_multipliers[i]  # 关节位置
                mirrored_obs[:, i + 12] = obs[:, i + 12] * joint_multipliers[i]  # 关节速度

    # 检查输出是否有效
    if torch.isnan(mirrored_actions).any() or torch.isinf(mirrored_actions).any():
        print("Warning: Output actions contain NaN or Inf values")
        return None, None

    if mirrored_obs is not None and (torch.isnan(mirrored_obs).any() or torch.isinf(mirrored_obs).any()):
        print("Warning: Output obs contain NaN or Inf values")
        return None, None

    # 打印输出统计信息
    try:
        if mirrored_obs is not None:
            print(
                f"Output obs - mean: {mirrored_obs.mean():.4f}, std: {mirrored_obs.std():.4f}, min: {mirrored_obs.min():.4f}, max: {mirrored_obs.max():.4f}"
            )
        print(
            f"Output actions - mean: {mirrored_actions.mean():.4f}, std: {mirrored_actions.std():.4f}, min: {mirrored_actions.min():.4f}, max: {mirrored_actions.max():.4f}"
        )
    except Exception as e:
        print(f"Warning: Error printing output statistics: {e}")

    print("=== End of Symmetry Augmentation ===\n")
    return mirrored_obs, mirrored_actions


@configclass
class D9RoughPPORunnerEasyCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "d9_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     # use_mirror_loss=True,
        #     # mirror_loss_coeff=0.001,  # 使用更小的系数
        #     data_augmentation_func=d9_leg_symmetry_augmentation
        # )
    )


@configclass
class D9FlatRunningPPORunnerEasyCfg(D9RoughPPORunnerEasyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=4.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

        self.max_iterations = 20000
        self.experiment_name = "d9_running"


@configclass
class D9FlatPPORunnerEasyCfg(D9RoughPPORunnerEasyCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "d9_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]
