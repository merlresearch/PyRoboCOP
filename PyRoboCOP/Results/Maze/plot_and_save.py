#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import matplotlib.pyplot as plt
import numpy as np


def parse_and_plot(ocp2nlpi, xopt, ocpi):
    """
    This function takes the solution object and ocl2nlp object
    """
    x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

    np.save("Results/Maze/traj_x_ring" + str(ocpi.ring) + ".npy", x_sol)
    np.save("Results/Maze/traj_xdot_ring" + str(ocpi.ring) + ".npy", xdot_sol)
    np.save("Results/Maze/control_ring" + str(ocpi.ring) + ".npy", u_sol)

    # plot diff vars
    state_traj = np.zeros((2, ocpi.T + 1))
    plt.figure()
    for i in range(2):
        tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, i, True)
        state_traj[i, :] = xarray
        if i == 0:
            plt.plot(tarray, xarray, label="theta")
        else:
            plt.plot(tarray, xarray, label="thetadot")
        plt.legend()

    # plot derivatives of diff vars
    plt.figure()
    for i in range(2):
        tarray, xarray = ocp2nlpi.extract_trajectory(xdot_sol, i, False)
        if i == 0:
            plt.plot(tarray, xarray, label="dthetadt")
        else:
            plt.plot(tarray, xarray, label="dthetadotdt")
        plt.legend()

    u_traj = np.zeros((ocpi.n_u, ocpi.T))
    plt.figure()
    for i in range(2):
        tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, i, False)
        u_traj[i, :] = xarray
        if i == 0:
            plt.plot(tarray, xarray, label="u beta")
        else:
            plt.plot(tarray, xarray, label="u gamma")
        plt.legend()

    plt.show()
