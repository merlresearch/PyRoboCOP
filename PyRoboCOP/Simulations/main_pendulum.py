#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.pendulum_adolcocp as ocp
import ipopt
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp
from Results.pendulum.plot_and_save import parse_and_plot

ocpi = ocp.invertedpendulumOCP()
ncolloc = 1
roots = "radau"  # "explicit"#
compl = 2
autodiff = 1
compladapt = 0
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
# nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption("tol", 1e-6)
# nlp.addOption('max_iter', 1)
# nlp.addOption('hessian_approximation', 'limited-memory')

"""
    Solve
"""
xopt, info = nlp.solve(x0_ipopt)
print(xopt)
# Parse and plot the solution
parse_and_plot(ocp2nlpi, xopt, ocpi)
