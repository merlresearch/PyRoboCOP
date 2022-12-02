#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import adolc as ad
import Envs.Dynamics.car2D2obstacle_adolcocp as ocp
import ipopt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytest
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp


def test_main_car2D2obstacle():
    ocpi = ocp.car2DobstacleOCP()
    ncolloc = 1
    roots = "radau"
    compl = 0
    autodiff = 1
    compladapt = 1
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
    nlp.addOption("max_iter", 10)
    # nlp.addOption('hessian_approximation', 'limited-memory')

    """
        Solve
    """
    xopt, info = nlp.solve(x0_ipopt)

    # Check by comparing actual values to expected. Cannot compare exactly due to floating point.
    # This is just a basic test. It would be better to check all outputs (eg using a numpy array).
    x_sol, xdot_sol, y_sol, u_sol = ocp2nlpi.parsesolution(xopt)
    tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
    tarray, yarray = ocp2nlpi.extract_trajectory(y_sol, 0, False)

    # print("xsol: ", np.shape(x_sol))
    # print("xsol: ", x_sol[-1, :, 0])

    expected_result = [2.0, 2.5, 1.57079633, 0.0]
    assert x_sol[-1, 0, 0] == pytest.approx(expected_result[0])
    assert x_sol[-1, 1, 0] == pytest.approx(expected_result[1])
    assert x_sol[-1, 2, 0] == pytest.approx(expected_result[2])
    assert x_sol[-1, 3, 0] == pytest.approx(expected_result[3])
