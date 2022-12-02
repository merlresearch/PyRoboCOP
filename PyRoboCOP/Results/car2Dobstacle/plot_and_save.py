#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch


def parse_and_plot(ocp2nlpi, xopt, ocpi):
    """
    This function takes the solution object and ocl2nlp object
    """

    # x_sol and u_sol are the output of the NLP

    x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

    np.save("Results/car2Dobstacle/traj.npy", x_sol)
    np.save("Results/car2Dobstacle/control.npy", u_sol)

    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
    tarray, yarray = ocp2nlpi.extract_trajectory(x_sol, 1, True)
    tarray, tharray = ocp2nlpi.extract_trajectory(x_sol, 2, True)
    plt.plot(xarray, yarray)
    ax = plt.gca()
    rect = patches.Rectangle((1.5, 1.0 + 0.15), 1.0, 1.0, angle=0.0, fill=True, color="r")
    ax.add_patch(rect)
    rect = patches.Rectangle((1.5, 3.0 - 0.15), 1.0, 1.0, angle=0.0, fill=True, color="r")
    ax.add_patch(rect)

    coordsA = "data"
    coordsB = "data"
    for v in range(4):
        vx = []
        vy = []

        alpha = [0, 0, 0, 0]
        alpha[v] = 1
        V = np.zeros((3, 4))
        for i in range(len(xarray)):
            ocpi.get_object_vertices(V, 1, [xarray[i], yarray[i], tharray[i], 0], [], [], [])
            vert1 = V * alpha
            vx.append(vert1[0])
            vy.append(vert1[1])
            # Connection lines
            for v in range(3):
                # ax.plot(V[0,v], V[1,v], marker='x')
                con = ConnectionPatch(
                    (V[0, v], V[1, v]),
                    (V[0, v + 1], V[1, v + 1]),
                    coordsA,
                    coordsB,
                    arrowstyle="-",
                    color="dodgerblue",
                )
                ax.add_artist(con)
            con = ConnectionPatch(
                (V[0, 3], V[1, 3]), (V[0, 0], V[1, 0]), coordsA, coordsB, arrowstyle="-", color="dodgerblue"
            )  # ,

            ax.add_artist(con)
        # ax.plot(vx, vy, marker='x')
    ax.axis([0.5, 3.0, 0.5, 4.5])
    ax.set_aspect("equal")
    ax.set_xlabel("X[m]")
    ax.set_ylabel("Y[m]")

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("filename.pdf", bbox_inches="tight", pad_inches=0)

    plt.savefig("Results/car2Dobstacle/carparking.pdf")

    plt.show()
