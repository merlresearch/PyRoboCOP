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
    x_sol, xdot_sol, y_sol, u_sol, params = ocp2nlpi.parsesolution(xopt)
    np.save("Results/Pushing_Mode/traj.npy", x_sol)

    np.save("Results/Pushing_Mode/control.npy", u_sol)

    print("Final time = ", params[0], params[1])

    plt.figure()

    ax = plt.gca()
    ax.axis([-0.2, 0.4, -0.2, 0.4])
    ax.set_aspect("equal", "box")

    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
    tarray, yarray = ocp2nlpi.extract_trajectory(x_sol, 1, True)
    tarray, tharray = ocp2nlpi.extract_trajectory(x_sol, 2, True)

    corners = ocpi.pushing_dynamics.corners()

    for i in range(int(len(xarray) / 3.0)):
        rc_trans = transform_points_twod(corners, np.array([xarray[3 * i], yarray[3 * i]]), tharray[3 * i])

        (rect,) = ax.plot(*rc_trans, color="dodgerblue")
        # ax.add_patch(rect)

    for i in range(int(len(xarray) / 3.0)):
        plt.arrow(
            xarray[3 * i], yarray[3 * i], 0.05 * np.cos(tharray[3 * i]), 0.05 * np.sin(tharray[3 * i]), color="orchid"
        )

    ax.set_ylabel("Y [m]")
    ax.set_xlabel("X [m]")

    plt.savefig("Results/Pushing_Mode/pushing_sequence.pdf")

    plt.figure()

    tarray1, uarray1 = ocp2nlpi.extract_trajectory(u_sol, 0, False)
    tabs = np.zeros_like(tarray1)
    for i in range(len(tarray1)):
        if tarray1[i] < 1.0:
            tabs[i] = params[0] * tarray1[i]
        else:
            tabs[i] = params[0] + params[1] * (tarray1[i] - 1.0)

    tarray2, uarray2 = ocp2nlpi.extract_trajectory(u_sol, 1, False)

    ax1 = plt.subplot(211)
    ax1.plot(tabs, uarray1, lw=2, color="dodgerblue")
    ax1.set_ylabel(r"$f_{\overrightarrow{n}}$ [N]")
    # ax1.set_xlabel('t [s]')

    ax2 = plt.subplot(212)
    ax2.plot(tabs, uarray2, lw=2, color="dodgerblue")
    ax2.set_ylabel(r"$f_{\overrightarrow{n}}$ [N]")
    ax2.set_xlabel("t [s]")

    plt.savefig("Results/Pushing_Mode/control_sequence.pdf")

    plt.show()
