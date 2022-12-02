#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import adolc as ad
import Envs.Dynamics.car2Dobstacle_adolcocp as ocp
import ipopt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import Solvers.ocp2nlp.OCP2NLPcompladolc as ocp2nlp

# from plotmanipulatormovement import VisualizeManipulator2DoF
ocpi = ocp.car2DobstacleOCP()
ncolloc = 1
roots = "radau"
# roots = "explicit"
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

tarray, xarray = ocp2nlpi.extract_trajectory(x_sol, 0, True)
tarray, yarray = ocp2nlpi.extract_trajectory(x_sol, 1, True)
tarray, tharray = ocp2nlpi.extract_trajectory(x_sol, 2, True)
plt.plot(xarray, yarray)
ax = plt.gca()
rect = patches.Rectangle((1.5, 1.5), 1.0, 1.0, angle=0.0, fill=True, color="r")
ax.add_patch(rect)

for v in range(4):
    vx = []
    vy = []
    alpha = [0, 0, 0, 0]
    alpha[v] = 1
    for i in range(len(xarray)):
        V = np.zeros((3, 4))
        ocpi.get_object_vertices(V, 1, [xarray[i], yarray[i], tharray[i], 0], [], [], [])
        vert1 = np.matmul(V, alpha)
        vx.append(vert1[0])
        vy.append(vert1[1])
    plt.plot(vx, vy, marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

"""
# plot diff vars
for i in range(ocpi.n_d):
    tarray,xarray = ocp2nlpi.extract_trajectory(x_sol,i,True)
    plt.plot(tarray,xarray)
    if i == 0:
        plt.ylabel("x")
    elif i == 1:
        plt.ylabel("y")
    elif i == 2:
        plt.ylabel("theta")
    else:
        plt.ylabel("v")
    plt.xlabel("t")
    plt.show()

# plot derivatives of diff vars
for i in range(ocpi.n_d):
    tarray,xarray = ocp2nlpi.extract_trajectory(xdot_sol,i,False)
    plt.plot(tarray,xarray)
    if i == 0:
        plt.ylabel("xdot")
    elif i == 1:
        plt.ylabel("ydot")
    elif i == 2:
        plt.ylabel("thetadot")
    else:
        plt.ylabel("vdot")
    plt.xlabel("t")
    plt.show()

# plot control vars
for i in range(ocpi.n_u):
    tarray,uarray = ocp2nlpi.extract_trajectory(u_sol,i,False)
    plt.plot(tarray,uarray)
    if i == 0:
      plt.ylabel("u_theta")
    else:
      plt.ylabel("u_v")
    plt.xlabel("t")
    plt.show()
"""
