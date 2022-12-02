#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np

"""
  Example 1 from:
  Quadratic Optimal Control of Linear Complementarity Systems: First order necessary conditions and numerical analysis
  Alexandre Vieira, Bernard Brogliato, and Christophe Prieur‡
  https://hal.inria.fr/hal-01690400v3/document
"""


class VBPexample1OCP(object):
    def __init__(self):
        self.a = 3.0
        self.b = -0.5
        self.d = 1.0
        self.e = -2.0
        self.f = 3.0
        self.tf = 1.0
        self.xic = 1.0

        self.n_d = 1
        self.n_a = 2
        self.n_u = 1
        self.n_p = 0
        self.n_cc = 1  # NEW
        self.T = 1000
        self.times = (self.tf / self.T) * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 7
        self.nnz_hess = 2

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
        if cc_bnd1 = 0, cc_bnd2 = 0
          x[cc_var1[i]]-lb[cc_var1[j]] >= 0 \perp x[cc_var2[i]]-lb[cc_var2[i]] >= 0
        if cc_bnd2 = 0, cc_bnd2 = 1
          x[cc_var1[i]]-lb[cc_var1[j]] >= 0 \perp ub[cc_var2[i]]-x[cc_var2[i]] >= 0
        if cc_bnd1 = 1, cc_bnd2 = 0
          lb[cc_var1[i]]-x[cc_var1[i]] >= 0 \perp x[cc_var2[i]]-lb[cc_var2[i]] >= 0
        if cc_bnd2 = 1, cc_bnd2 = 1
          ub[cc_var2[i]]-x[cc_var1[i]] >= 0 \perp ub[cc_var2[i]]-x[cc_var2[i]] >= 0
        """
        cc_var1 = np.array([0])
        cc_bnd1 = np.array([0])
        cc_var2 = np.array([1])
        cc_bnd2 = np.array([0])
        return cc_var1, cc_bnd1, cc_var2, cc_bnd2

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        # NEW
        lb = np.array([-1.0e30, -1.0e30, 0.0, 0.0, -1.0e30])
        ub = np.array([1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30])
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([0])
        xdot0 = np.array([0])
        u0 = np.array([0.0])
        y0 = np.array([0, 0.0])
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
        c = x[0] ** 2 + u[0] ** 2
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
        g[4] = 2 * u[0]
        # return g

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control variables at time t
        params - parameters
        """
        # c = np.zeros(self.n_d+self.n_a-self.n_cc)
        # c = c.astype(object)
        c[0] = -xdot[0] + self.a * x[0] + self.b * y[0] + self.f * u[0]
        c[1] = -y[1] + self.d * y[0] + self.e * u[0]
        # return c

    def jacobianstructure(self, row, col):
        row = [0, 0, 0, 0, 1, 1, 1]
        col = [0, 1, 2, 4, 2, 3, 4]
        # return row, col

    def jacobian(self, jac, t, x, xdot, y, u, params):
        """
        Method to return the jacobian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control variables at time t
        params - parameters
        """
        jac = [self.a, -1.0, self.b, self.f, self.d, -1.0, self.e]
        # return np.array(jac)

    def hessianstructure(self, row, col):
        row = np.array([0, 4])
        col = np.array([0, 4])
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
        # hesst = np.zeros((2,1))
        hesst[0] = 2 * obj_factor
        hesst[1] = 2 * obj_factor
        # return hesst
