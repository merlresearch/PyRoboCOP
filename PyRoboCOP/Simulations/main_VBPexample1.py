#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.VBPexample1OCP as ocp
import ipopt
import matplotlib.pyplot as plt
import numpy as np
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp

# from plotmanipulatormovement import VisualizeManipulator2DoF
ocpi = ocp.VBPexample1OCP()
ncolloc = 1
roots = "radau"
compl = 0
compladapt = 1
autodiff = 1
ocp2nlpi = ocp2nlp.OCP2NLP(ocpi, ncolloc, roots, compl, compladapt=compladapt, autodiff=autodiff)

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
for i in range(1):
    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, i, True)
    plt.plot(tarray, xarray)
    plt.ylabel("x")
    plt.show()

# plot derivatives of diff vars
for i in range(1):
    tarray, xarray = ocp2nlpi.extract_trajectory(xdot_sol, i, False)
    plt.plot(tarray, xarray)
    plt.ylabel("dxdt")
    plt.show()

# plot derivatives of complementarity vars
tarray, xarray0 = ocp2nlpi.extract_trajectory(y_sol, 0, False)
tarray, xarray1 = ocp2nlpi.extract_trajectory(y_sol, 1, False)
plt.plot(tarray, xarray0, tarray, xarray1)
plt.legend(["v", "vperp"])
plt.show()

tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 0, False)
plt.plot(tarray, xarray)
plt.ylabel("u")
# plt.show()
