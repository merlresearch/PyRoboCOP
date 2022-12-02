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
    for i in range(ocpi.n_d):
        tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, i, True)
        plt.figure()
        plt.plot(tarray, xarray)
        if i == 0:
            plt.ylabel("x")
            np.save("Results/pendulum/q_pendulum_pyrobocop.npy", xarray)
        elif i == 1:
            plt.ylabel("xdot")
        plt.xlabel("t")
        plt.savefig("Results/pendulum/state" + str(i))

    plt.figure()
    tarray, uarray = ocp2nlpi.extract_trajectory(u_sol, 0, False)
    np.save("Results/pendulum/u_pendulum_pyrobocop.npy", uarray)
    plt.plot(tarray, uarray)
    plt.ylabel("tau")
    plt.grid()
    plt.savefig("Results/pendulum/control.pdf")
    plt.show()
