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

    np.save("Results/Cartpole/traj.npy", x_sol)

    np.save("Results/Cartpole/control.npy", u_sol)

    plt.figure()
    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
    plt.plot(xarray)
    plt.savefig("Results/Cartpole/q_0_cartpole.pdf")

    plt.figure()
    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 1, True)
    plt.plot(xarray)
    plt.savefig("Results/Cartpole/q_1_cartpole.pdf")

    plt.figure()

    tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 0, False)
    plt.plot(tarray, xarray)
    plt.ylabel("tau")
    plt.savefig("Results/Cartpole/u_cartpole.pdf")
    plt.show()
