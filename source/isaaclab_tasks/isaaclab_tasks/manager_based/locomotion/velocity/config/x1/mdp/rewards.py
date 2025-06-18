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

def leg_arm_symmetric(
        env: ManagerBasedRLEnv, 
        asset_cfg_leg: SceneEntityCfg = SceneEntityCfg("robot"),
        asset_cfg_arm: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one using L2 norm."""
    # extract the used quantities (to enable type-hinting)
    asset_leg: Articulation = env.scene[asset_cfg_leg.name]
    asset_arm: Articulation = env.scene[asset_cfg_arm.name]

    leg_vel = asset_leg.data.joint_vel[:, asset_cfg_leg.joint_ids]
    arm_vel = asset_arm.data.joint_vel[:, asset_cfg_arm.joint_ids]
    angle = leg_vel - arm_vel

    return torch.norm(angle, p=2, dim=1)