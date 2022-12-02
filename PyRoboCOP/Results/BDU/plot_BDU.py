# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"family": "normal", "size": 12}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Load Data
# Trajectory keypoint 1
xyz_kp1_subtask1 = np.load("traj_subtask1_scenario123_x.npy")[:, :3, 1]
xyz_kp1_subtask2 = np.load("traj_subtask2_scenario1_x.npy")[:, :3, 1]
# Trajectory keypoint 2
xyz_kp2_subtask1 = np.load("traj_subtask1_scenario123_x.npy")[:, 3:6, 1]
xyz_kp2_subtask2 = np.load("traj_subtask2_scenario1_x.npy")[:, 3:6, 1]
# Trajectory complementarity variables
y_subtask1 = np.load("traj_subtask1_scenario123_y.npy")
y_subtask2 = np.load("traj_subtask2_scenario1_y.npy")
# Trajectory control
u_subtask1 = np.load("traj_subtask1_scenario123_control.npy")
u_subtask2 = np.load("traj_subtask2_scenario1_control.npy")


# ---- XYZ trajectory plots
x_sol_k1 = np.vstack((xyz_kp1_subtask1, xyz_kp1_subtask2))
x_sol_k2 = np.vstack((xyz_kp2_subtask1, xyz_kp2_subtask2))


def change_ref_frame(x_sol):
    # match the coordinates in simulation and OCP
    new_x = x_sol[:, 0:3] - np.array([0.55, 0.0, 0.45])
    new_x[:, [0, 1]] = new_x[:, [1, 0]]
    new_x[:, 1] = -new_x[:, 1]
    return new_x


new_x_sol_k1 = change_ref_frame(x_sol_k1)
new_x_sol_k2 = change_ref_frame(x_sol_k2)

# Figures
plt.figure()
plt.plot(new_x_sol_k1[:, 0], "b")
plt.plot(new_x_sol_k1[:, 1], "r--")
plt.plot(new_x_sol_k1[:, 2], "g-.")
plt.legend(["x", "y", "z"])
plt.savefig("traj_xyz_k1_BDU.pdf")

plt.figure()
plt.plot(new_x_sol_k2[:, 0], "b")
plt.plot(new_x_sol_k2[:, 1], "r--")
plt.plot(new_x_sol_k2[:, 2], "g-.")
plt.legend(["x", "y", "z"])
plt.savefig("traj_xyz_k2_BDU.pdf")

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Points
ax.scatter(new_x_sol_k1[::2, 0], new_x_sol_k1[::2, 1], new_x_sol_k1[::2, 2], color="orange", marker="o")
ax.scatter(new_x_sol_k2[::2, 0], new_x_sol_k2[::2, 1], new_x_sol_k2[::2, 2], color="b", marker="o")
# Virtual connections
for k1x, k1y, k1z, k2x, k2y, k2z, in zip(
    new_x_sol_k1[::2, 0],
    new_x_sol_k1[::2, 1],
    new_x_sol_k1[::2, 2],
    new_x_sol_k2[::2, 0],
    new_x_sol_k2[::2, 1],
    new_x_sol_k2[::2, 2],
):
    ax.plot([k1x, k2x], [k1y, k2y], [k1z, k2z], color="grey", marker=".")
# initial point
ax.scatter(new_x_sol_k1[0, 0], new_x_sol_k1[0, 1], new_x_sol_k1[0, 2], color="g", marker="o", linewidth=4)
ax.scatter(new_x_sol_k2[0, 0], new_x_sol_k2[0, 1], new_x_sol_k2[0, 2], color="g", marker="o", linewidth=4)
# Final point
ax.scatter(new_x_sol_k1[-1, 0], new_x_sol_k1[-1, 1], new_x_sol_k1[-1, 2], color="r", marker="o", linewidth=4)
ax.scatter(new_x_sol_k2[-1, 0], new_x_sol_k2[-1, 1], new_x_sol_k2[-1, 2], color="r", marker="o", linewidth=4)

plt.legend()

plt.show()
