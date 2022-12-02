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

    np.save("Results/BDU/traj_subtask1_scenario" + str(ocpi.scenario) + "_x", x_sol)
    np.save("Results/BDU/traj_subtask1_scenario" + str(ocpi.scenario) + "_xdot", xdot_sol)
    np.save("Results/BDU/traj_subtask1_scenario" + str(ocpi.scenario) + "_control.npy", u_sol)
    np.save("Results/BDU/traj_subtask1_scenario" + str(ocpi.scenario) + "_y.npy", y_sol)

    plt.figure()
    plt.title("pos_vel")
    for i in range(12):
        tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, i, True)
        plt.subplot(2, 6, i + 1)
        plt.plot(tarray, xarray)

    plt.figure()
    tarray, xarray0 = ocp2nlpi.extract_trajectory(y_sol, 0, False)
    tarray, xarray2 = ocp2nlpi.extract_trajectory(y_sol, 2, False)
    plt.plot(tarray, xarray0, label="y0")
    plt.plot(tarray, xarray2, label="y2")
    plt.legend()

    plt.figure()
    tarray, xarray1 = ocp2nlpi.extract_trajectory(y_sol, 1, False)
    tarray, xarray3 = ocp2nlpi.extract_trajectory(y_sol, 3, False)
    plt.plot(tarray, xarray1, label="y1")
    plt.plot(tarray, xarray3, label="y3")
    plt.legend()

    plt.figure()
    tarray, xarray4 = ocp2nlpi.extract_trajectory(y_sol, 4, False)
    plt.plot(tarray, xarray4, label="y4")
    plt.legend()

    plt.figure()
    tarray, xarray1 = ocp2nlpi.extract_trajectory(x_sol, 1, True)
    tarray, xarray2 = ocp2nlpi.extract_trajectory(x_sol, 2, True)
    plt.plot(xarray1, xarray2, label="y_and_z")

    plt.figure()
    plt.title("u")
    for i in range(ocpi.n_u):
        tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, i, False)
        plt.subplot(1, 3, i + 1)
        plt.plot(tarray, xarray)
    plt.show()
