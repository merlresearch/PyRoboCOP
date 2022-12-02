#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.quadrotor_adolcocp as ocp
import ipopt
import matplotlib.pyplot as plt
import numpy as np
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp
from Envs.Dynamics.param_dict_quadrotor import param_dict

# from plotmanipulatormovement import VisualizeManipulator2DoF
ocpi = ocp.quadrotor_ocp(param_dict)
ncolloc = 1
roots = "radau"  # "legendre"
compl = 0
compladapt = 0
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
nlp.addOption("mu_strategy", "monotone")
nlp.addOption("tol", 1e-6)
# nlp.addOption('max_iter', 1)
# nlp.addOption('hessian_approximation', 'limited-memory')

"""
    Solve
"""
xopt, info = nlp.solve(x0_ipopt)

# x_sol and u_sol are the output of the NLP

x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

plt.figure()
tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
plt.plot(xarray)

plt.figure()
tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 1, True)
plt.plot(xarray)

plt.figure()
tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 2, True)
plt.plot(xarray)

plt.figure()
tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 0, False)
plt.plot(tarray, xarray)
plt.ylabel("tau")
plt.figure()

tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 1, False)
plt.plot(tarray, xarray)
plt.ylabel("tau")
plt.figure()

tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 2, False)
plt.plot(tarray, xarray)
plt.ylabel("tau")
plt.figure()

tarray, xarray = ocp2nlpi.extract_trajectory(u_sol, 3, False)
plt.plot(tarray, xarray)
plt.ylabel("tau")
plt.show()
