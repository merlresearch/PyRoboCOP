#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""

Description: Main file to compute the optimal trajectory for subtask 2 of the Belt Drive Unit
"""


import Envs.Dynamics.BDU_subtask2_adolc as ocp
import ipopt
import matplotlib.pyplot as plt

# See ICRA paper for explanation of the different scenarios. Default scenario is 1
import numpy as np
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp

scenario = 1  # See ICRA paper for explanation of the different scenarios. Default scenario is 1
ocpi = ocp.loop_2_kps_OCP(scenario)
ncolloc = 1
roots = "radau"  # "legendre"
compl = 2
compladapt = 0
autodiff = 0
ocp2nlpi = ocp2nlp.OCP2NLP(ocpi, ncolloc, roots, compl=compl, compladapt=compladapt, autodiff=autodiff)

ocp2nlpi.print_ocp()

"""
    Create the Ipopt Problem Instance
"""
x0_ipopt = ocp2nlpi.initialpoint()
lb_ipopt, ub_ipopt, lbcon_ipopt, ubcon_ipopt = ocp2nlpi.bounds()
nlp = ipopt.problem(
    n=ocp2nlpi.n_ipopt,
    m=ocp2nlpi.m_ipopt,
    problem_obj=ocp2nlpi,
    lb=lb_ipopt,
    ub=ub_ipopt,
    cl=lbcon_ipopt,
    cu=ubcon_ipopt,
)
"""
    Set solver options
"""
# nlp.addOption('derivative_test', 'first-order')
# nlp.addOption('derivative_test', 'second-order')
nlp.addOption("mu_strategy", "monotone")
nlp.addOption("tol", 1e-5)
# nlp.addOption('mumps_mem_percent', 5)
nlp.addOption("max_iter", 3000)
# nlp.addOption("mu_superlinear_decrease_power",1.2)
nlp.addOption("hessian_approximation", "limited-memory")  # don't use hessian
"""
    Solve
"""
xopt, info = nlp.solve(x0_ipopt)
x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)


np.save("Results/BDU/traj_subtask2_scenario" + str(ocpi.scenario) + "_x", x_sol)
np.save("Results/BDU/traj_subtask2_scenario" + str(ocpi.scenario) + "_xdot", xdot_sol)
np.save("Results/BDU/traj_subtask2_scenario" + str(ocpi.scenario) + "_control.npy", u_sol)
np.save("Results/BDU/traj_subtask2_scenario" + str(ocpi.scenario) + "_y.npy", y_sol)

# For the plots shown in the paper see Results/BDU/plot_BDU.py

plt.figure()
plt.title("state_pos_vel_ang")
for i in range(18):
    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, i, True)
    plt.subplot(3, 6, i + 1)
    plt.plot(tarray, xarray)

plt.figure()

tarray, xarray0 = ocp2nlpi.extract_trajectory(y_sol, 0, False)
tarray, xarray1 = ocp2nlpi.extract_trajectory(y_sol, 1, False)
plt.plot(tarray, xarray0, label="y0")
plt.plot(tarray, xarray1, label="y1")
plt.legend()

plt.figure()
tarray, xarray1 = ocp2nlpi.extract_trajectory(y_sol, 2, False)
plt.plot(tarray, xarray1, label="y2")
plt.legend()

plt.figure()
tarray, xarray0 = ocp2nlpi.extract_trajectory(x_sol, 0, True)
tarray, xarray2 = ocp2nlpi.extract_trajectory(x_sol, 2, True)
plt.plot(xarray0, xarray2, label="x1_vs_z1")
plt.legend()

plt.figure()
tarray, xarray0 = ocp2nlpi.extract_trajectory(x_sol, 0, True)
tarray, xarray2 = ocp2nlpi.extract_trajectory(x_sol, 2, True)
plt.plot(((xarray0 - 0.55) ** 2 + (xarray2 - 0.37) ** 2) ** 0.5, label="length")
plt.legend()

plt.figure()
plt.title("u")
for i in range(ocpi.n_u):
    tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, i, False)
    plt.subplot(1, 6, i + 1)
    plt.plot(tarray, xarray)
plt.show()
