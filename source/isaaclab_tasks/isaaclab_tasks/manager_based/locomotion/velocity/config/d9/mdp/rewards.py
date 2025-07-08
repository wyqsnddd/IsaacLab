from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def utils_no_fly(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty: return 1 if exactly one foot is in contact (no flying), else 0."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]
    contacts = forces > 0.1
    single = torch.sum(contacts.float(), dim=1) == 1
    return single.float()

def joint_deviation_l2(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one using L2 norm."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # compute L2 norm (Euclidean norm)
    return torch.norm(angle, p=2, dim=1)

def alternating_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
    std: float = 0.4,
    force_threshold: float = 10.0,
    double_air_time_weight: float = 0.2,  # 新增：双脚腾空的权重
) -> torch.Tensor:
    """奖励交替腾空的步态模式，同时给予双脚腾空的次要奖励。

    这个函数结合了相位控制和接触检测，确保机器人实现稳定的交替腾空步态。
    它奖励：
    1. 正确的交替腾空模式（主要奖励）
    2. 合适的空中时间
    3. 与期望相位的同步
    4. 短暂的双脚腾空（次要奖励）

    Args:
        env: RL环境实例
        sensor_cfg: 接触传感器的配置
        period: 步态周期（秒）
        std: 指数核的标准差
        force_threshold: 接触力的阈值
        double_air_time_weight: 双脚腾空奖励的权重

    Returns:
        奖励值
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 1. 计算当前相位
    dt = env.step_dt
    phase = (env.episode_length_buf * dt) % period / period

    # 2. 获取接触状态
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    in_contact = torch.norm(contact_forces, dim=2) > force_threshold
    in_contact_float = in_contact.float()

    # 3. 获取空中时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # 4. 计算期望的接触状态（基于相位）
    desired_contact_left = (phase > 0.5).float()
    desired_contact_right = (phase <= 0.5).float()

    # 5. 计算接触状态误差
    contact_error_left = torch.square(in_contact_float[:, 0] - desired_contact_left)
    contact_error_right = torch.square(in_contact_float[:, 1] - desired_contact_right)

    # 6. 计算空中时间奖励
    air_time_reward_left = torch.where(
        desired_contact_left == 0,
        torch.clamp(air_time[:, 0] / period, max=0.5),
        torch.clamp(1.0 - air_time[:, 0] / period, min=0.5),
    )

    air_time_reward_right = torch.where(
        desired_contact_right == 0,
        torch.clamp(air_time[:, 1] / period, max=0.5),
        torch.clamp(1.0 - air_time[:, 1] / period, min=0.5),
    )

    # 7. 计算交替性奖励
    alternation_reward = torch.where(torch.abs(in_contact_float[:, 0] - in_contact_float[:, 1]) > 0.5, 1.0, 0.0)

    # 8. 新增：计算双脚腾空奖励
    both_feet_air = (in_contact_float[:, 0] == 0) & (in_contact_float[:, 1] == 0)
    min_air_time = torch.min(air_time[:, 0], air_time[:, 1])

    # 双脚腾空奖励：基于最短的空中时间，但限制在较小范围内
    double_air_time_reward = torch.where(
        both_feet_air, torch.clamp(min_air_time / period, max=0.2), 0.0  # 限制最大奖励为周期的20%
    )

    # 9. 组合所有奖励
    phase_reward = torch.exp(-(contact_error_left + contact_error_right) / std)
    time_reward = (air_time_reward_left + air_time_reward_right) / 2.0

    # 主要奖励：交替步态
    main_reward = phase_reward * time_reward * alternation_reward

    # 次要奖励：双脚腾空
    secondary_reward = double_air_time_reward * double_air_time_weight

    # 10. 最终奖励：主要奖励 + 次要奖励
    final_reward = main_reward + secondary_reward

    # 11. 只在有速度命令时给予奖励
    command = env.command_manager.get_command("base_velocity")
    velocity_threshold = 0.1
    has_velocity = torch.norm(command[:, :2], dim=1) > velocity_threshold

    return final_reward * has_velocity.float()

def kicking_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    friction_coefficient: float = 0.7,
    std: float = 0.25,
) -> torch.Tensor:
    """Penalize contact forces that violate the friction cone constraint.
    
    This function penalizes tangential contact forces that are too large relative to the normal force,
    which would indicate slipping or improper foot-ground interaction. Forces are properly decomposed
    using projection operators relative to the gravity direction.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        friction_coefficient: The coefficient of friction between the foot and ground.
        std: Standard deviation for the exponential kernel to smooth the penalty.
        
    Returns:
        A negative reward (penalty) that is larger when friction cone violations are more severe.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces for all feet
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    
    # Define gravity direction (assuming z-up coordinate system)
    gravity_dir = torch.tensor([0.0, 0.0, 1.0], device=contact_forces.device)
    
    # Project forces onto gravity direction to get normal components
    # P = vv^T where v is the normalized gravity direction
    gravity_dir_normal = gravity_dir / torch.norm(gravity_dir)
    projection_matrix = torch.outer(gravity_dir_normal, gravity_dir_normal)
    
    # Reshape forces for batch matrix multiplication
    batch_size = contact_forces.shape[0]
    num_feet = contact_forces.shape[1]
    forces_reshaped = contact_forces.reshape(-1, 3)
    
    # Calculate normal forces using projection
    normal_forces = torch.matmul(forces_reshaped, projection_matrix)
    normal_forces = normal_forces.reshape(batch_size, num_feet, 3)
    
    # Calculate tangential forces using null-space projection
    # I - P gives us the null-space projector
    null_space_matrix = torch.eye(3, device=contact_forces.device) - projection_matrix
    tangential_forces = torch.matmul(forces_reshaped, null_space_matrix)
    tangential_forces = tangential_forces.reshape(batch_size, num_feet, 3)
    
    # Get magnitudes
    normal_magnitude = torch.norm(normal_forces, dim=-1)
    tangential_magnitude = torch.norm(tangential_forces, dim=-1)
    
    # Calculate the maximum allowed tangential force based on friction cone
    max_tangential = friction_coefficient * torch.abs(normal_magnitude)
    
    # Calculate how much the tangential force exceeds the friction cone
    violation = torch.clamp(tangential_magnitude - max_tangential, min=0.0)
    
    # Apply exponential penalty to smooth the transition
    penalty = torch.exp(-violation / std)
    
    # Only apply penalty when there is actual contact (normal force > 0)
    mask = normal_magnitude > 0.0
    penalty = torch.where(mask, penalty, torch.ones_like(penalty))
    
    # Return negative reward (penalty)
    return -torch.mean(penalty, dim=1)

def energy_efficiency_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励能量效率"""
    asset = env.scene[asset_cfg.name]
    # 计算关节力矩和速度的乘积（功率）
    power = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)
    # 使用指数核函数将功率映射到奖励
    return torch.exp(-power / 100.0)  # 100.0是缩放因子