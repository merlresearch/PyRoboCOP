#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Description: Main file to compute the optimal trajectory for subtask 1 of the Belt Drive Unit
"""


import Envs.Dynamics.BDU_subtask1_adolc as ocp
import ipopt
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp
from Results.BDU.plot_and_save import parse_and_plot

ocpi = ocp.loop_2_kps_OCP()
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
# nlp.addOption('max_iter', 1)
# nlp.addOption("mu_superlinear_decrease_power",1.2)
# nlp.addOption('hessian_approximation', 'limited-memory')
"""
    Solve
"""
xopt, info = nlp.solve(x0_ipopt)
x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

# Parse and plot the solution. Simple plot of all the optimal variables
# For the plots shown in the paper see Results/BDU/plot_BDU.py
parse_and_plot(ocp2nlpi, xopt, ocpi)
