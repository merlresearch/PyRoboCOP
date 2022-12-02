#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time

import Envs.Dynamics.maze as ocp
import ipopt
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp
from Results.Maze.plot_and_save import parse_and_plot

# ----- ocp
# Maze Env init
ring = 1
ocpi = ocp.Maze(ring)
# reset the state
ocpi.initialcondition()

ncolloc = 1
roots = "radau"  # "explicit" # "legendre"
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
# nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption("tol", 1e-6)
# nlp.addOption('max_iter', 500)
# nlp.addOption('hessian_approximation', 'limited-memory')
"""
    Solve
"""
st_time = time.time()
xopt, info = nlp.solve(x0_ipopt)
print("Total time to solve problem ", time.time() - st_time)
print("Time spent in adolc = ", ocp2nlpi.time_adolc)
# x_sol and u_sol are the output of the NLP
x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

# Parse and plot the solution. Simple plot of all the optimal variables
# For the plots shown in the paper see Results/Maze/plot_maze.py
parse_and_plot(ocp2nlpi, xopt, ocpi)
