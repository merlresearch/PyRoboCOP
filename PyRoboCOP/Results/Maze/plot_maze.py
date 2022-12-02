# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Envs.Dynamics.CircularMaze.MazePlot2D import VisualizeMaze

font = {"family": "normal", "size": 15}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Pyrobocop
path = "Results/Maze/"
x_maze = np.load(path + "traj_x_ring1.npy")[:, :, 1]
l_ring = len(x_maze[:, 0])
control_maze = np.load(path + "control_ring1.npy")[:, :, 0]

for i in range(1, 4):
    x_maze = np.vstack((x_maze, np.load(path + "traj_x_ring" + str(i) + ".npy")[:, :, 1]))
    control_maze = np.vstack((control_maze, np.load(path + "control_ring" + str(i) + ".npy")[:, :, 0]))

gate = namedtuple("gate", ["radiuses", "phi"])
ring_radius = [0.09, 0.07, 0.05, 0.03, 0]
ring_radius = [9, 7, 5, 3, 0]

gates = [
    gate([ring_radius[4], ring_radius[3]], 0),
    gate([ring_radius[3], ring_radius[2]], np.pi / 4),
    gate([ring_radius[3], ring_radius[2]], np.pi / 4 + np.pi / 2),
    gate([ring_radius[3], ring_radius[2]], np.pi / 4 + np.pi),
    gate([ring_radius[3], ring_radius[2]], np.pi / 4 + 1.5 * np.pi),
    gate([ring_radius[2], ring_radius[1]], 0),
    gate([ring_radius[2], ring_radius[1]], 0 + np.pi / 2),
    gate([ring_radius[2], ring_radius[1]], 0 + np.pi),
    gate([ring_radius[2], ring_radius[1]], 0 + 1.5 * np.pi),
    gate([ring_radius[1], ring_radius[0]], np.pi / 4),
    gate([ring_radius[1], ring_radius[0]], np.pi / 4 + np.pi / 2),
    gate([ring_radius[1], ring_radius[0]], np.pi / 4 + np.pi),
    gate([ring_radius[1], ring_radius[0]], np.pi / 4 + 1.5 * np.pi),
]

radius = np.concatenate(
    (
        np.array([ring_radius[0]] * l_ring),
        np.array([ring_radius[1]] * l_ring),
        np.array([ring_radius[2]] * l_ring),
        np.array([ring_radius[3]] * l_ring),
    )
)
ds = 2
vis = VisualizeMaze(theta=x_maze[::ds, 0], radius=radius[::ds], gates=gates, ring_radius=ring_radius)
# print ('.. Showing Video ..')
# vis.show_animation(save_filename='havij') #save_filename='havij')
vis.plot_traj(int(l_ring * 4 / ds))
plt.savefig("maze_traj.pdf")

fig, axs = plt.subplots(4)
# fig.suptitle('Optimal control sequences')

axs[0].plot(np.load(path + "control_ring1.npy")[:, 0, 0], "b", label="motor1")
axs[0].plot(np.load(path + "control_ring1.npy")[:, 1, 0], "r", label="motor2")
# axs[0].set_title('Ring 1')
axs[0].legend(("motor 1", "motor 2"), loc="lower right", shadow=False)
axs[0].set_ylabel("[rad]")

axs[1].plot(np.load(path + "control_ring2.npy")[:, 0, 0], "b", label="motor1")
axs[1].plot(np.load(path + "control_ring2.npy")[:, 1, 0], "r", label="motor2")
# axs[1].set_title('Ring 2')
axs[1].legend(("motor 1", "motor 2"), loc="lower right", shadow=False)
axs[1].set_ylabel("[rad]")

axs[2].plot(np.load(path + "control_ring3.npy")[:, 0, 0], "b", label="motor1")
axs[2].plot(np.load(path + "control_ring3.npy")[:, 1, 0], "r", label="motor2")
# axs[2].set_title('Ring 3')
axs[2].legend(("motor 1", "motor 2"), loc="lower right", shadow=False)
axs[2].set_ylabel("[rad]")

axs[3].plot(np.load(path + "control_ring4.npy")[:, 0, 0], "b", label="motor1")
axs[3].plot(np.load(path + "control_ring4.npy")[:, 1, 0], "r", label="motor2")
# axs[3].set_title('Ring 4')
axs[3].legend(("motor 1", "motor 2"), loc="lower right", shadow=False)
axs[3].set_ylabel("[rad]")

# axs.legend()


plt.show()
