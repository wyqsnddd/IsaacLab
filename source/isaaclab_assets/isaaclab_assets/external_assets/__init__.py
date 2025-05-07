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
