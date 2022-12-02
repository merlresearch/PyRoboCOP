#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.pendulum_softwalls_ocp as ocp
import ipopt
import matplotlib.pyplot as plt
import numpy as np
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp

ocpi = ocp.invertedpendulumOCP()
ncolloc = 1

roots = "radau"
roots = "explicit"
compl = 2
compladapt = 1
ocp2nlpi = ocp2nlp.OCP2NLP(ocpi, ncolloc, roots, compl, compladapt, autodiff=0)

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
# nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption("tol", 1e-6)
# nlp.addOption('max_iter', 1)
# nlp.addOption('hessian_approximation', 'limited-memory')

"""
    Solve
"""
xopt, info = nlp.solve(x0_ipopt)

# x_sol and u_sol are the output of the NLP

x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

np.save("traj.npy", x_sol)
np.save("control.npy", u_sol)

# plot diff vars
plt.figure()
for i in range(2):
    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, i, True)
    plt.plot(tarray, xarray)
    if i == 0:
        plt.ylabel("theta")
    else:
        plt.ylabel("thetadot")
    # plt.show()

# plot derivatives of diff vars
plt.figure()
for i in range(2):
    tarray, xarray = ocp2nlpi.extract_trajectory(xdot_sol, i, False)
    plt.plot(tarray, xarray)
    if i == 0:
        plt.ylabel("dthetadt")
    else:
        plt.ylabel("dthetadotdt")
    # plt.show()

# plot derivatives of complementarity vars
plt.figure()
tarray, xarray0 = ocp2nlpi.extract_trajectory(y_sol, 0, False)
tarray, xarray1 = ocp2nlpi.extract_trajectory(y_sol, 2, False)
plt.plot(tarray, xarray0, tarray, xarray1)
plt.legend(["v0", "vperp0"])
# plt.show()

plt.figure()
tarray, xarray0 = ocp2nlpi.extract_trajectory(y_sol, 1, False)
tarray, xarray1 = ocp2nlpi.extract_trajectory(y_sol, 3, False)
plt.plot(tarray, xarray0, tarray, xarray1)
plt.legend(["v1", "vperp1"])
# plt.show()

plt.figure()
tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 0, False)
plt.plot(tarray, xarray)
plt.ylabel("tau")
plt.show()
