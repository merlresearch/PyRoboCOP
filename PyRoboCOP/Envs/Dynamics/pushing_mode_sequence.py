#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np
from Envs.Dynamics.planar_pushing import HybridPlanarPusher


class planar_pushing_mode_dependent(object):
    def __init__(self):
        self.n_d = 3
        self.n_a = 2
        self.n_u = 2
        self.n_p = 2
        self.n_cc = 0
        self.T = 200
        self.times = (2.0 / self.T) * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 0
        self.nnz_hess = 0

        self.dt = 0.25
        self.mu = 0.3

        self.pushing_dynamics = HybridPlanarPusher(self.dt, self.mu)

    def get_info(self):
        """
        Method to return OCP info
        n_d - number of differential vars
        n_a - number of algebraic vars
        n_u - number or controls vars
        n_cc - number of complementarity variables (part of algebraic vars)
        n_p - number of parameters
        T   - number of time-steps
        times - the time at start of each of the time intervals, an array of (T+1)
        nnz_jac - number of nonzeros in jacobian of DAE
        nnz_hess - number of nonzeros in hessian of OCP at each time-step
        """
        return self.n_d, self.n_a, self.n_u, self.n_p, self.n_cc, self.T, self.times, self.nnz_jac, self.nnz_hess

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        # Lower bounds of the state variables.
        xlb = np.array([-5.0, -5.0, -2 * np.pi])
        xub = np.array([5.0, 5.0, 2 * np.pi])
        # Bounds for the state derivatives
        xdotlb = np.array([-5, -5, -5])
        xdotub = np.array([5, 5, 5])
        # Bounds of algebraic variable
        ylb = np.array([0.0, -np.infty])
        yub = np.array([np.infty, 0.0])
        # Bounds of control inputs
        ulb = np.array([0, -1.0])
        uub = np.array([0.5, 1.0])

        # Stack them to get the ub and lb vector
        lb = np.hstack((xlb, xdotlb, ylb, ulb))
        ub = np.hstack((xub, xdotub, yub, uub))

        return lb, ub

    def bounds_params(self):
        """
        Method to return the bounds on the parameters in the OCP instance
        lbp - lower bound array
        ubp - upper bound array
        """
        return np.array([0.01, 0.01]), np.array([100.0, 200.0])

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([0, 0, 0])
        xdot0 = np.array([0, 0, 0])
        y0 = np.array([0.0, 0.0])
        u0 = np.array([0, 0])
        return x0, xdot0, y0, u0

    def initialpoint_params(self):
        """
        Method to return the initial guess for the parameters in the OCP instance

        """
        return np.array([10.0, 20.0])

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = [0, 0, 0]
        return xic

    def bounds_finaltime(self):
        """
        Method to return the bounds on the states at final time
        """
        lbxf = np.array([0.0, 0.0, np.pi])
        ubxf = np.array([0.0, 0.0, np.pi])
        return lbxf, ubxf

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        return 0.0

    def objective_mayer(self, x0, xf, params):
        """
        Method to return the objective function of the OCP instance that is not integral
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # DR: should we have instead only one obejctive function with a flag?
        tf1 = params[0]
        tf2 = params[1]
        return tf1 + tf2

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # c = np.zeros((self.n_d+self.n_a,1))
        # c = c.astype(object)
        tf1 = params[0]
        tf2 = params[1]

        if t <= 1.0:
            _, dxdt = self.pushing_dynamics.dynamics(x, u, 0)
            c[0] = -xdot[0] + tf1 * dxdt[0]
            c[1] = -xdot[1] + tf1 * dxdt[1]
            c[2] = -xdot[2] + tf1 * dxdt[2]

        else:
            _, dxdt = self.pushing_dynamics.dynamics(x, u, 1)
            c[0] = -xdot[0] + tf2 * dxdt[0]
            c[1] = -xdot[1] + tf2 * dxdt[1]
            c[2] = -xdot[2] + tf2 * dxdt[2]

        # # algebraic constraints for friction cone
        c[3] = y[0] - u[1] - self.mu * u[0]
        c[4] = y[1] - u[1] + self.mu * u[0]
        # return c
