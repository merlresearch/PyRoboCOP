#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from Envs.Dynamics.rotations import transform_points_twod
from matplotlib.patches import Rectangle


def parse_and_plot(ocp2nlpi, xopt, ocpi):
    """
    This function takes the solution object and ocl2nlp object
    """
    x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

    np.save("Results/Pushing_Slipping_Obstacles/traj.npy", x_sol)
    np.save("Results/Pushing_Slipping_Obstacles/algebraic_variables.npy", y_sol)
    np.save("Results/Pushing_Slipping_Obstacles/control.npy", u_sol)

    plt.figure()

    ax = plt.gca()
    ax.axis([-0.2, 0.7, -0.2, 0.7])
    ax.set_xlabel("X[m]")
    ax.set_ylabel("Y[m]")
    ax.set_aspect("equal", "box")
    rect = patches.Rectangle((0.45, 0.57), 0.1, 0.1, angle=0.0, fill=True, color="r")
    ax.add_patch(rect)
    rect = patches.Rectangle((0.45, 0.33), 0.1, 0.1, angle=0.0, fill=True, color="r")
    ax.add_patch(rect)

    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
    tarray, yarray = ocp2nlpi.extract_trajectory(x_sol, 1, True)
    tarray, tharray = ocp2nlpi.extract_trajectory(x_sol, 2, True)

    corners = ocpi.pushing_dynamics.corners()

    for i in range(int(len(xarray) / 2.0)):
        rc_trans = transform_points_twod(corners, np.array([xarray[2 * i], yarray[2 * i]]), tharray[2 * i])

        (rect,) = ax.plot(*rc_trans, color="dodgerblue")
        # ax.add_patch(rect)

    for i in range(int(len(xarray) / 2.0)):
        plt.arrow(
            xarray[2 * i], yarray[2 * i], 0.05 * np.cos(tharray[2 * i]), 0.05 * np.sin(tharray[2 * i]), color="orchid"
        )

    plt.savefig("Results/Pushing_Slipping/pushing_sequence.pdf")
    tarray, yarray1 = ocp2nlpi.extract_trajectory(y_sol, 2, False)

    tarray, yarray2 = ocp2nlpi.extract_trajectory(y_sol, 3, False)

    vel = yarray1 - yarray2
    tarray1, uarray1 = ocp2nlpi.extract_trajectory(u_sol, 0, False)

    tarray2, uarray2 = ocp2nlpi.extract_trajectory(u_sol, 1, False)

    plt.figure()
    ax1 = plt.subplot(311)
    ax1.plot(tarray1, uarray1, lw=2, color="dodgerblue")
    ax1.set_ylabel(r"$f_{\overrightarrow{n}}$ [N]")
    # ax1.set_xlabel('t [s]')

    ax2 = plt.subplot(312)
    ax2.plot(tarray2, uarray2, lw=2, color="dodgerblue")
    ax2.set_ylabel(r"$f_{\overrightarrow{t}}$ [N]")
    # ax2.set_xlabel('t [s]')

    ax3 = plt.subplot(313)
    ax3.plot(tarray, vel * 0.09, lw=2, color="dodgerblue")
    ax3.set_ylabel(r"$\dot{p_y}$[m/s]")
    ax3.set_xlabel("t [s]")

    plt.savefig("Results/Pushing_Slipping/control_sequence.pdf")

    plt.show()
