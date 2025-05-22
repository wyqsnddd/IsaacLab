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


def phase_based_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
    offset: float = 0.5,
    std: float = 0.1,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward for phase-based contact pattern that encourages periodic foot contacts.

    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        period: Period of the gait cycle in seconds.
        offset: Phase offset between left and right feet (0.5 for alternating).
        std: Standard deviation for the exponential kernel.
        force_threshold: Threshold for considering a contact as active.

    Returns:
        The reward value.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Calculate phases using environment's step_dt
    dt = env.step_dt  # Get dt from environment's step_dt property
    phase = (env.episode_length_buf * dt) % period / period
    phase_left = phase
    phase_right = (phase + offset) % 1.0

    # Get contact states
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    in_contact = torch.norm(contact_forces, dim=2) > force_threshold

    # Convert boolean contact states to float
    in_contact_float = in_contact.float()

    # Calculate desired contact states based on phase
    # We want contact when phase is between 0.5 and 1.0
    desired_contact_left = (phase_left > 0.5).float()
    desired_contact_right = (phase_right > 0.5).float()

    # Calculate contact errors
    contact_error_left = torch.square(in_contact_float[:, 0] - desired_contact_left)
    contact_error_right = torch.square(in_contact_float[:, 1] - desired_contact_right)

    # Calculate reward using exponential kernel
    reward_left = torch.exp(-contact_error_left / std)
    reward_right = torch.exp(-contact_error_right / std)

    # Combine rewards
    return (reward_left + reward_right) / 2.0


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
    air_time_diff = torch.clip(air_time_diff, max=max_err ** 2)
    contact_time_diff = torch.clip(contact_time_diff, max=max_err ** 2)

    # Calculate reward using exponential kernel
    sync_reward = torch.exp(-(air_time_diff + contact_time_diff) / std)

    # Only enable reward when moving
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), sync_reward, 0.0)


def utils_no_fly(
    env: ManagerBasedRLEnv,  # W: Trailing whitespace
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty: return 1 if exactly one foot is in contact (no flying), else 0."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]
    contacts = forces > 0.1
    single = torch.sum(contacts.float(), dim=1) == 1
    return single.float()


def contact_no_velocity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contact with no velocity to prevent sliding or stalling.

    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.
        force_threshold: Threshold for considering a contact as active.

    Returns:
        The reward value (negative penalty).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # Get contact forces and foot velocities
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    foot_velocities = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :3]

    # Check which feet are in contact (force magnitude > threshold)
    in_contact = torch.norm(contact_forces, dim=2) > force_threshold

    # Get velocities of feet that are in contact
    contact_feet_vel = foot_velocities * in_contact.unsqueeze(-1)

    # Calculate penalty as squared velocity of contacting feet
    penalty = torch.square(contact_feet_vel)

    # Sum over all feet and dimensions
    return -torch.sum(penalty, dim=(1, 2))
