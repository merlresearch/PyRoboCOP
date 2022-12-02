#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np


class invertedpendulumOCP(object):
    def __init__(self):

        self.param_dict = {"m": 1, "l": 1, "b": 0.01, "dt": 0.05, "g": 9.81}

        self.objchoice = 2

        self.n_d = 2
        self.n_a = 0
        self.n_u = 1
        self.n_p = 0
        self.n_cc = 0
        self.T = 150
        self.times = self.param_dict["dt"] * np.array(list(range(self.T + 1)))
        self.nnz_jac = 0
        self.nnz_hess = 0

    def get_model_params(self):
        return self.param_dict["m"], self.param_dict["l"], self.param_dict["b"], self.param_dict["g"]

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
        return self.n_d, self.n_a, self.n_u, self.n_p, self.n_cc, self.T, self.times, self.nnz_jac, self.nnz_hess

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        lb = np.array([-2 * np.pi, -10.0, -1.0e30, -1.0e30, -10])
        ub = np.array([2 * np.pi, 10.0, 1.0e30, 1.0e30, 10])
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([0, 0.0])
        xdot0 = np.array([0, 0.0])
        u0 = np.array([0.0])
        y0 = np.array([])
        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.array([0.0, 0.0])
        return xic

    # def bounds_finaltime(self):
    #     """
    #     Method to return the bounds on final time state for the OCP instance
    #     """
    #     xfc = np.array([np.pi, 0.0])  # np.pi/2
    #     return xfc, xfc

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        m, l, b, g = self.get_model_params()
        theta = x[0]
        thetadot = x[1]
        c = (theta - np.pi) ** 2 + thetadot**2 + 0.01 * u[0] ** 2
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

        theta = x[0]
        thetadot = x[1]
        dthetadt = xdot[0]
        dthetadotdt = xdot[1]
        m, l, b, g = self.get_model_params()
        I = m * g * l**2
        c[0] = -dthetadt + thetadot
        c[1] = -dthetadotdt * I + (u[0] - m * g * l * np.sin(theta) - b * thetadot)
