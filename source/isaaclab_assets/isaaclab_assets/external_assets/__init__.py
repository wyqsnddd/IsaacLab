# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

# Register UI extensions.
from .ui_extension_example import *

# Parent Path
RootPath = os.path.dirname(os.path.abspath(__file__))
AssetPath = os.path.join(RootPath, "assets")
ActuatorPath = os.path.join(RootPath, "actuators")


import os
from pathlib import Path

# 定义基础路径
RootPath = Path(__file__).parent
ExternalAssetPath = os.path.join(RootPath, "assets")
ExternalActuatorPath = os.path.join(RootPath, "actuators")


class ExternalAssetLoader:
    """Asset loader class for managing asset paths and loading."""

    def __init__(self):
        self.asset_path = ExternalAssetPath
        self.actuator_path = ExternalActuatorPath
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required paths exist."""
        if not os.path.exists(self.asset_path):
            raise FileNotFoundError(f"Asset path not found: {self.asset_path}")
        if not os.path.exists(self.actuator_path):
            raise FileNotFoundError(f"Actuator path not found: {self.actuator_path}")

    def get_asset_path(self, asset_name):
        """Get full path for an asset."""
        return os.path.join(self.asset_path, asset_name)

    def get_actuator_path(self, actuator_name):
        """Get full path for an actuator."""
        return os.path.join(self.actuator_path, actuator_name)

    def get_robot_usd_path(self, robot_name, usd_file):
        """Get path for robot USD file."""
        return os.path.join(self.asset_path, "usd_files", "robots", robot_name, usd_file)

    @property
    def root_path(self):
        """Get the root path."""
        return RootPath


# 创建默认实例
external_asset_loader = ExternalAssetLoader()

# 导出需要的变量和类
__all__ = ["ExternalAssetPath", "ExternalActuatorPath", "ExternalAssetLoader", "external_asset_loader"]
