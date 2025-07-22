import h5py
from tqdm import tqdm
import numpy as np
import random
import os
import sys
import mujoco.viewer
import mujoco
import time

if __name__ == "__main__":
    xml_path = "/home/eric/project/lab_repo/source/isaaclab_assets/isaaclab_assets/external_assets/assets/usd_files/robots/pudu_d9/scene.xml"   
    simulation_duration = 60.0
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002
    m.opt.gravity = [0, 0, 0]

    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        print(i, ":", jname)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 5.0  # Increasing this value can pull the view further away
        viewer.cam.azimuth = 90    # Horizontal rotation angle (degrees)
        viewer.cam.elevation = -15 # Elevation angle (negative values indicate looking down)
        viewer.cam.lookat[:] = [0, 0, 0.5]  # Point the camera at a specific point in the coordinate system

        d.qpos[0] = 0
        d.qpos[1] = 0
        d.qpos[2] = 0.88
        d.qpos[3:7] = [1,0,0,0]
        d.qpos[7 : 7+15] = [-0., -0., 0.3, -0.7, 0.4, 0, 0., 0., 0.3, -0.7, 0.4, 0, 0, 0.2, 0.2]

        d.qvel[:] = 0
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(50)