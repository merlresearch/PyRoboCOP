#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np
from Envs.Dynamics.quadrotor_dynamics import quadrotor_dyn


class quadrotor_ocp(object):
    def __init__(self, params):

        self.params = params
        self.n_d = 12
        self.n_a = 0
        self.n_u = 4
        self.n_p = 0
        self.n_cc = 0  # NEW
        self.T = 200
        self.dt = 0.033
        self.times = self.dt * np.array(list(range(self.T + 1)))
        self.nnz_jac = 0
        self.nnz_hess = 0

        # Import the dynamics class
        self.quadrotor = quadrotor_dyn(self.params)

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

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        # NEW
        lbx = -np.Infinity * np.ones(self.n_d)
        ubx = np.Infinity * np.ones(self.n_d)
        lbxdot = -np.Infinity * np.ones(self.n_d)
        ubxdot = np.Infinity * np.ones(self.n_d)
        lbu = [-10, -10, -10, -10]
        ubu = [10, 10, 10, 10]
        lb = np.hstack([lbx, lbxdot, lbu])
        ub = np.hstack([ubx, ubxdot, ubu])
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """

        x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        xdot0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        u0 = np.array([0, 0, 0, 0])
        y0 = []
        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return xic

    def bounds_finaltime(self):
        """
        Method to return the bounds on final time state for the OCP instance
        """
        xfc = np.array([2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return xfc, xfc

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        c = (x[0] - 2.0) * (x[0] - 2.0)
        c = c + (x[1] - 2.0) * (x[1] - 2.0)
        c = c + (x[2] - 3) * (x[2] - 3)
        c = c + u[0] * u[0]
        c = c + u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
        return c

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # c = np.zeros(self.n_d+self.n_a,)
        # c = c.astype(object)
        dxdt = self.quadrotor.get_dynamics_equations(x, u)

        for i in range(self.n_d):
            c[i] = -xdot[i] + dxdt[i]

        # return c
