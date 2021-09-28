# %%
# import
import os
import sys

import time
import numpy as np
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import Joint
from abr_control.interfaces.mujoco import Mujoco

from abr_control_mod.mujoco_utils import (
    get_rope_body_ids, get_body_center_of_mass, 
    apply_impulse_com_batch)
from common.urscript_control_util import get_movej_trajectory
from scipy.interpolate import interp1d
# from matplotlib import pyplot as plt


# %%
dt = 0.001
sim_duration = 4.0
subsample_rate = 10

acceleration = 10.0
speed = 3.0
j0_goal = np.pi/2*1.5
j1_goal = np.pi/2*1.5

xml_dir = os.path.join(
    os.path.dirname(__file__), 'assets', 'rope_2d')
xml_fname = 'double_joint_rope.xml'


num_sim_steps = int(sim_duration / dt)
j_start = np.array([0,0])
j_end = np.array([j0_goal, j1_goal])

q_target = get_movej_trajectory(
    j_start=j_start, j_end=j_end, 
    acceleration=acceleration, speed=speed, dt=dt)
qdot_target = np.gradient(q_target, dt, axis=0)

# %%
robot_config = MujocoConfig(
    xml_file=xml_fname, 
    folder=xml_dir)
interface = Mujoco(robot_config, dt=dt, visualize=True)
interface.connect()
ctrlr = Joint(robot_config, kp=3000)

interface.set_joint_state(q=j_start, dq=np.zeros(2))
rope_body_ids = get_rope_body_ids(interface.sim.model)

rope_history = list()
for i in range(num_sim_steps):
    feedback = interface.get_feedback()

    idx = min(i, len(q_target)-1)
    u = ctrlr.generate(
        q=feedback['q'],
        dq=feedback['dq'],
        target=q_target[idx],
        target_velocity=qdot_target[idx]
    )
    
    if i % subsample_rate == 0:
        rope_body_com = get_body_center_of_mass(
            interface.sim.data, rope_body_ids)
        rope_history.append(rope_body_com)
    interface.send_forces(u)

# %%
interface.disconnect()
