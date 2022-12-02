#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np

r"""
  Example 3 from:
  Quadratic Optimal Control of Linear Complementarity Systems: First order necessary conditions and numerical analysis
  Alexandre Vieira, Bernard Brogliato, and Christophe Prieur‡
  https://hal.inria.fr/hal-01690400v3/document

  ## System dynamics   Equation:
  \dot{x} = A x(t)+B v(t) +C u(t), x \in R 2
  0 \leq v(t) \perp   D x(t) + v(t) +  E u(t) \geq 0
  x(0)= [-1/2, 1]^T, x(t) free
"""


class VBPexample3OCP(object):
    def __init__(self):
        self.tf = 1.0
        self.xic = np.array([-0.5, 1.0])

        self.n_d = 2
        self.n_a = 4
        self.n_u = 2
        self.n_p = 0
        self.n_cc = 2  # NEW
        self.T = 1000
        self.times = (self.tf / self.T) * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 25
        self.nnz_hess = 4

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

    def get_complementarity_info(self):
        r"""
        Method to return the complementarity info
        cc_var1   - index of complementarity variables (>= 0, < n_a)
        cc_bnd1   - 0/1 array 0: lower bound, 1: upper bound
        cc_var2   - index of complementarity variables (>= 0, < n_a)
        cc_bnd2   - 0/1 array 0: lower bound, 1: upper bound
        complemntarity constraints for i = 1,...,n_cc
        y : algebraic variables
        if cc_bnd1 = 0, cc_bnd2 = 0
          y[cc_var1[i]]-lb[cc_var1[i]] >= 0 \perp y[cc_var2[i]]-lb[cc_var2[i]] >= 0
        if cc_bnd2 = 0, cc_bnd2 = 1
          y[cc_var1[i]]-lb[cc_var1[i]] >= 0 \perp ub[cc_var2[i]]-y[cc_var2[i]] >= 0
        if cc_bnd1 = 1, cc_bnd2 = 0
          ub[cc_var1[i]]-x[cc_var1[i]] >= 0 \perp y[cc_var2[i]]-lb[cc_var2[i]] >= 0
        if cc_bnd2 = 1, cc_bnd2 = 1
          ub[cc_var2[i]]-y[cc_var1[i]] >= 0 \perp ub[cc_var2[i]]-y
          [cc_var2[i]] >= 0
        """
        cc_var1 = np.reshape([0, 1], (2, 1))
        cc_bnd1 = np.array([0, 0])
        cc_var2 = np.reshape([2, 3], (2, 1))
        cc_bnd2 = np.array([0, 0])
        return cc_var1, cc_bnd1, cc_var2, cc_bnd2

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        # NEW
        lb = np.array([-1.0e30, -1.0e30, -1.0e30, -1.0e30, 0.0, 0.0, 0.0, 0.0, -1.0e30, -1.0e30])
        ub = np.array([1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30])
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([0, 0])
        # NEW
        xdot0 = np.array([0, 0])
        u0 = np.array([0.0, 0.0])
        y0 = np.array([0, 0.0, 0.0, 0.0])
        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        return self.xic

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        c = x[0] ** 2 + x[1] ** 2 + 25 * u[0] ** 2 + 25 * u[1] ** 2
        return c

    def gradient(self, g, t, x, xdot, y, u, params):
        """
        Method to return the gradient of the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # g = np.zeros((2*self.n_d + self.n_a + self.n_u,1))
        g = 0.0 * g
        g[0] = 2 * x[0]
        g[1] = 2 * x[1]
        g[8] = 50 * u[0]
        g[9] = 50 * u[1]
        # return g

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # c = np.zeros(self.n_d+self.n_a-self.n_cc)
        # c = c.astype(object)
        c[0] = -xdot[0] + x[0] + 2 * x[1] - y[0] + y[1] + u[0] + 3 * u[1]
        c[1] = -xdot[1] + 2 * x[0] + x[1] - y[0] + y[1] + 2 * u[0] + u[1]
        c[2] = -y[2] + 3 * x[0] - x[1] + y[0] + u[0] - u[1]
        c[3] = -y[3] - 2 * x[0] + y[1] - u[0] + 2 * u[1]
        # return c

    def jacobianstructure(self, row, col):
        # no complementarity constraints here, ocp2nlp takes care of it
        row = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        col = [0, 1, 2, 4, 5, 8, 9, 0, 1, 3, 4, 5, 8, 9, 0, 1, 4, 6, 8, 9, 0, 5, 7, 8, 9]
        # return row, col

    def jacobian(self, jac, t, x, xdot, y, u, params):
        """
        Method to return the jacobian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        jac = [
            1.0,
            2.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            3.0,
            2.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            2.0,
            1.0,
            3.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -2.0,
            1.0,
            -1.0,
            -1.0,
            2.0,
        ]
        # return np.array(jac)

    def hessianstructure(self, row, col):
        row = np.array([0, 1, 8, 9])
        col = np.array([0, 1, 8, 9])
        # return row, col

    def hessian(self, hesst, t, x, xdot, y, u, params, mult, obj_factor):
        """
        Method to return the hessian of the Lagrangian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # hesst = np.zeros((4,1))
        hesst[0] = 2 * obj_factor
        hesst[1] = 2 * obj_factor
        hesst[2] = 50 * obj_factor
        hesst[3] = 50 * obj_factor
        # return hesst
