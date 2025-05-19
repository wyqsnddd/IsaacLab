# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for D9's locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Penalize variance in air time between feet to encourage consistent gait.

    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        std: Standard deviation for the exponential kernel.

    Returns:
        The reward value.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # Get air times for all feet
    air_times = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    # Calculate variance across feet
    mean_air_time = torch.mean(air_times, dim=1, keepdim=True)
    variance = torch.mean(torch.square(air_times - mean_air_time), dim=1)

    # Convert variance to reward using exponential kernel
    return torch.exp(-variance / std)


def biped_gait_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float,
    max_err: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward for bipedal walking gait that encourages alternating foot contacts.

    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.
        std: Standard deviation for the exponential kernel.
        max_err: Maximum error threshold for clipping.
        velocity_threshold: Velocity threshold for enabling the reward.

    Returns:
        The reward value.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # Get air times and contact times for both feet
    air_times = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_times = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # Calculate synchronization error between feet
    air_time_diff = torch.square(air_times[:, 0] - air_times[:, 1])
    contact_time_diff = torch.square(contact_times[:, 0] - contact_times[:, 1])

    # Clip errors
    air_time_diff = torch.clip(air_time_diff, max=max_err**2)
    contact_time_diff = torch.clip(contact_time_diff, max=max_err**2)

    # Calculate reward using exponential kernel
    sync_reward = torch.exp(-(air_time_diff + contact_time_diff) / std)

    # Only enable reward when moving
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), sync_reward, 0.0)
