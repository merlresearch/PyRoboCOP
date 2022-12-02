#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.VBPexample1OCP as ocp
import ipopt
import numpy as np
import pytest
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp


def test_main_VBPexample1():
    ocpi = ocp.VBPexample1OCP()
    ncolloc = 1
    roots = "radau"
    compl = 0
    compladapt = 1
    autodiff = 0
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
    nlp.addOption("max_iter", 10)
    # nlp.addOption('hessian_approximation', 'limited-memory')

    """
        Solve
    """
    xopt, info = nlp.solve(x0_ipopt)

    # Check by comparing actual values to expected. Cannot compare exactly due to floating point.
    # This is just a basic test. It would be better to check all outputs (eg using a numpy array).
    x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)

    print("xsol: ", np.shape(x_sol))
    print("xsol: ", x_sol[-1, :, 0])

    expected_result = [0.40621075]
    assert x_sol[-1, 0, 0] == pytest.approx(expected_result[0])
