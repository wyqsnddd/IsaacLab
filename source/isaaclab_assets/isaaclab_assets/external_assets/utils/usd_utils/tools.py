# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdLux, UsdPhysics  # noqa F401


def create_fixed_joint2(joint_path, prim0_path):
    """
    it works, make sure the prim path is right.
    """
    stage = get_current_stage()
    d6FixedJoint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    d6FixedJoint.GetBody1Rel().SetTargets([prim0_path])  # the body should be 1

    # joint_prim = get_prim_at_path(joint_path)
    # PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    return


def fix_to_ground(joint_path, prim_path, anchor_pos=Gf.Vec3f(0, 0, 0)):
    create_fixed_joint2(joint_path, prim_path)
    # create_fixed_joint(joint_path,
    #                    prim0_path="/World/defaultGroundPlane",
    #                    prim1_path=prim_path,
    #                    pos0=anchor_pos,
    #                    rot0=Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)),
    #                    pos1=Gf.Vec3f(0, 0, 0.18),
    #                    rot1=Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
