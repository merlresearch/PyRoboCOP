#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Non-linear dynamics for cartpole.
The equations of motion are derived in Russ Tedrake's course on underactuated dynamics.
It can be found here:
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/Tedrake-Aug09.pdf

m_c -- mass of the cart
m_p -- mass of the pendulum
l -- length of pendulum
\\theta -- angle of the pendulum from the stable equilibrium
x -- position of the cart

The equations of motion are derived in the standard form :

H(q)\\ddot{q} +C(q,\\dot{q})+G(q) = Bu

where q= [x,\\theta]^T
"""


import copy

import numpy as np
from Envs.Dynamics.param_dict_cartpole import param_dict
from numpy import cos, sin


class CartPole(object):
    def __init__(self):

        self.param_dict = param_dict

        # Next we list down all the parameters of the cartpole dynamical system

        self.mc = param_dict["mc"]
        self.mp = param_dict["mp"]
        self.l = param_dict["l"]
        self.dt = param_dict["dt"]
        self.g = self.param_dict["g"]

        self.n_d = 4
        self.n_a = 0
        self.n_cc = 0
        self.n_u = 1
        self.n_p = 0
        self.T = 100
        self.times = self.dt * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 0
        self.nnz_hess = 0

    def get_info(self):
        """
        Method to return OCP info
        n_d - number of differential vars
        n_a - number of algebraic vars
        n_u - number or controls vars
        n_p - number of parameters
        n_cc - number of complementarity variables (part of algebraic vars)
        T   - number of time-steps
        times - the time at start of each of the time intervals, an array of (T+1)
        nnz_jac - number of nonzeros in jacobian of DAE
        nnz_hess - number of nonzeros in hessian of OCP at each time-step
        """
        # NEW
        return self.n_d, self.n_a, self.n_u, self.n_p, self.n_cc, self.T, self.times, self.nnz_jac, self.nnz_hess

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([3, 0, 0, 0])
        # NEW
        xdot0 = np.array([0, 0, 0, 0])
        u0 = np.array([5])
        y0 = np.array([0])

        return x0, xdot0, y0, u0

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        # NEW
        lb = np.array([-5, -2 * np.pi, -10.0, -10, -10, -10, -100, -100, -20])
        ub = np.array([5, 2 * np.pi, 10.0, 10, 10, 10, 100, 100, 20])
        return lb, ub

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.array([3.0, 0.0, 0.0, 0.0])
        return xic

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """

        theta = x[1]
        c = (theta - np.pi) ** 2 + 0.1 * x[2] ** 2 + 0.1 * x[3] ** 2 + 0.1 * (x[0] - 0.5) ** 2 + 0.01 * u[0] ** 2
        return c

    def bounds_finaltime(self):
        """
        Method to return the bounds on final time state for the OCP instance
        """
        xfc = np.array([0.5, np.pi, 0.0, 0.0])
        return xfc, xfc

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """

        xx = x[0]
        theta = x[1]
        xxdot = x[2]
        thetadot = x[3]

        dxxdt = xdot[0]
        dthetadt = xdot[1]
        dxxdotdt = xdot[2]
        dthetadotdt = xdot[3]

        tau = u[0]

        # Next we introduce the dynamics constraint as imposed by the underlying dynamics of the cartpole system.
        c[0] = -xdot[0] + x[2]
        c[1] = -xdot[1] + x[3]

        c[2] = -xdot[2] + (tau + self.mp * sin(theta) * (self.l * thetadot**2 + self.g * cos(theta))) / (
            self.mc + self.mp * sin(theta) ** 2
        )

        c[3] = -xdot[3] + (
            -tau * cos(theta)
            - self.mp * self.l * thetadot**2 * cos(theta) * sin(theta)
            - (self.mc + self.mp) * self.g * sin(theta)
        ) / (self.l * (self.mc + self.mp * sin(theta) ** 2))
