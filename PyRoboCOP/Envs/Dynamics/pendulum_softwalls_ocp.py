#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np


class invertedpendulumOCP(object):
    def __init__(self):

        self.param_dict = {
            "k1": 1.0e5,
            "k2": 1.0e5,
            "m": 0.1,
            "l": 0.5,
            "b": 0.0,
            "dt": 0.05,
            "g": 9.81,
            "d": 0.1,
        }

        self.objchoice = 1

        self.n_d = 2
        self.n_a = 4
        self.n_u = 1
        self.n_p = 0
        self.n_cc = 2
        self.T = 150
        self.times = self.param_dict["dt"] * np.array(list(range(self.T + 1)))
        self.nnz_jac = 14
        self.nnz_hess = 3
        # Complementarity variables
        self.cc_var1 = np.reshape([0, 1], (2, 1))
        self.cc_bnd1 = np.array([0, 0])
        self.cc_var2 = np.reshape([2, 3], (2, 1))
        self.cc_bnd2 = np.array([0, 0])

    def get_model_params(self):
        return (
            self.param_dict["m"],
            self.param_dict["l"],
            self.param_dict["b"],
            self.param_dict["g"],
            self.param_dict["k1"],
            self.param_dict["k2"],
            self.param_dict["d"],
        )

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

    def get_complementarity_info(self):

        return self.cc_var1, self.cc_bnd1, self.cc_var2, self.cc_bnd2

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        lb = np.array([-2 * np.pi, -10.0, -1.0e30, -1.0e30, 0.0, 0.0, 0.0, 0.0, -10])
        ub = np.array([2 * np.pi, 10.0, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 10])
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([np.pi - 0.2, 0.0])
        xdot0 = np.array([0, 0.0])
        u0 = np.array([2.0])
        y0 = np.array([0.0, 0.0, 0.0, 0.0])
        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.array([np.pi - 0.2, 0.0])
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
        m, l, b, g, k1, k2, d = self.get_model_params()
        theta = x[0]
        thetadot = x[1]
        if self.objchoice == 1:
            c = (theta - np.pi) ** 2 + thetadot**2 + 0.01 * u[0] ** 2
        else:
            c = (l * np.sin(theta)) ** 2 + (l * np.cos(theta) + l) ** 2 + thetadot**2 + 0.01 * u**2
        return c

    def gradient(self, grad, t, x, xdot, y, u, params):
        """
        Method to return the gradient of the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        m, l, b, g, k1, k2, d = self.get_model_params()
        grad[:, 0] = 0.0
        theta = x[0]
        thetadot = x[1]
        if self.objchoice == 1:
            grad[0] = 2 * (theta - np.pi)
        else:
            grad[0] = 2 * l**2 * np.sin(theta) * np.cos(theta) - 2 * l**2 * (np.cos(theta) + 1) * np.sin(theta)
            grad[0] = -2 * l**2 * np.sin(theta)
        grad[1] = 2 * thetadot
        grad[8] = 0.02 * u[0]

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
        m, l, b, g, k1, k2, d = self.get_model_params()
        I = m * g * l**2
        c[0] = -dthetadt + thetadot
        c[1] = -I * dthetadotdt + (u[0] - m * g * l * np.sin(theta) - b * thetadot) + y[0] / (l * m) - y[1] / (l * m)
        # complementarity
        c[2] = -y[2] + l * np.sin(theta) + y[0] / k1 + d
        c[3] = -y[3] - l * np.sin(theta) + y[1] / k2 + d

    def jacobianstructure(self, row, col):

        row[:, 0] = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        col[:, 0] = [1, 2, 0, 1, 3, 4, 5, 8, 0, 4, 6, 0, 5, 7]

    def jacobian(self, jac, t, x, xdot, y, u, params):
        """
        Method to return the jacobian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        theta = x[0]
        m, l, b, g, k1, k2, d = self.get_model_params()
        I = m * g * l**2
        jac[0, 0] = 1.0
        jac[1, 0] = -1.0
        jac[2, 0] = -m * g * l * np.cos(theta)
        jac[3, 0] = -b
        jac[4, 0] = -I
        jac[5, 0] = 1 / (l * m)
        jac[6, 0] = -1 / (l * m)
        jac[7, 0] = 1.0
        jac[8, 0] = l * np.cos(theta)
        jac[9, 0] = 1 / k1
        jac[10, 0] = -1
        jac[11, 0] = -l * np.cos(theta)
        jac[12, 0] = 1 / k1
        jac[13, 0] = -1

    def hessianstructure(self, row, col):
        row[:, 0] = np.array([0, 1, 8])
        col[:, 0] = np.array([0, 1, 8])

    def hessian(self, hesst, t, x, xdot, y, u, params, mult, obj_factor):
        """
        Method to return the hessian of the Lagrangian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        theta = x[0]
        m, l, b, g, k1, k2, d = self.get_model_params()
        if self.objchoice == 1:
            hesst[0] = 2 * obj_factor
        else:
            hesst[0] = -2 * l**2 * np.cos(theta) * obj_factor
        if len(mult) > 0:
            hesst[0] = (
                hesst[0]
                + mult[1] * (m * g * l * np.sin(theta))
                - mult[2] * l * np.sin(theta)
                + mult[3] * l * np.sin(theta)
            )
        hesst[1] = 2 * obj_factor
        hesst[2] = 0.02 * obj_factor
