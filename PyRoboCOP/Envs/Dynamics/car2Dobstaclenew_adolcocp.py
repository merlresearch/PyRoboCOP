#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np


class car2DobstacleOCP(object):
    def __init__(self):

        self.n_d = 4
        self.n_a = 0
        self.n_u = 2
        self.n_p = 0
        self.n_cc = 0
        self.T = 50
        self.times = 0.1 * np.array(list(range(self.T + 1)))
        self.nnz_jac = 0
        self.nnz_hess = 0

        self.qc = [2, 2, 0]
        self.r = 0.5
        self.W = np.eye(3)

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
        lbx = -np.Infinity * np.ones(self.n_d)
        ubx = np.Infinity * np.ones(self.n_d)
        lbxdot = -np.Infinity * np.ones(self.n_d)
        ubxdot = np.Infinity * np.ones(self.n_d)
        lbu = [-np.pi / 3, -10]
        ubu = [np.pi / 3, 10]
        lb = np.hstack([lbx, lbxdot, lbu])
        ub = np.hstack([ubx, ubxdot, ubu])
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        alpha = (self.times[self.T - 1] - t) / self.times[self.T - 1]
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        x0[0] = 2.0 - 2 * np.sin(np.pi / 2 * alpha)
        x0[1] = 0 + 2 * np.cos(np.pi / 2 * alpha)
        x0[3] = (1 - alpha) * np.pi / 3
        xdot0 = np.array([x0[3] * np.cos(x0[2]), x0[3] * np.sin(x0[2]), 0.0, 0.0])
        u0 = np.array([2.0, 5])
        y0 = []
        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.array([0.0, 0.0, 0.0, 0.0])
        return xic

    def bounds_finaltime(self):
        """
        Method to return the final time state bounds for the OCP instance
        """
        xfc = np.array([3.0, 3.0, np.pi / 2, 0.0])
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
        c = (x[0] - 3.0) * (x[0] - 3.0)
        c = c + (x[1] - 3.0) * (x[1] - 3.0)
        c = c + (x[2] - np.pi / 3) * (x[2] - np.pi / 3)
        c = c + x[3] * x[3]
        c = c + u[0] * u[0]
        c = c + u[1] * u[1]
        return c

    def constraint(self, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        c = np.zeros(
            self.n_d + self.n_a,
        )
        c = c.astype(object)
        c[0] = -xdot[0] + x[3] * np.sin(x[2])
        c[1] = -xdot[1] + x[3] * np.cos(x[2])
        c[2] = -xdot[2] + u[0] * x[3]
        c[3] = -xdot[3] + u[1]
        return c

    def get_num_obstacles_controlled_objects(self):
        return 1, 1

    def get_staticObstacleInfo(self, ind):
        if ind == 0:
            return self.qc, self.W, self.r

    def get_controlledObject_number_vertices(self, ind):
        nv = 0
        if ind == 0:
            nv = 4
        return nv

    def get_controlledObject_convex_combination(self, ind, x, xdot, y, u, alpha):
        if ind == 0:
            nv = 4
            verts = np.zeros((3, nv))
            verts = verts.astype(object)
            sidexby2 = 0.5
            sideyby2 = 0.5
            diagby2 = np.sqrt(sidexby2**2 + sideyby2**2)
            x_c = x[0]
            y_c = x[1]
            theta = x[2]
            verts[0, 0] = x_c + diagby2 * np.cos(np.pi / 4 - theta)
            verts[1, 0] = y_c + diagby2 * np.sin(np.pi / 4 - theta)
            verts[0, 1] = x_c + diagby2 * np.cos(3 * np.pi / 4 - theta)
            verts[1, 1] = y_c + diagby2 * np.sin(3 * np.pi / 4 - theta)
            verts[0, 2] = x_c + diagby2 * np.cos(5 * np.pi / 4 - theta)
            verts[1, 2] = y_c + diagby2 * np.sin(5 * np.pi / 4 - theta)
            verts[0, 3] = x_c + diagby2 * np.cos(7 * np.pi / 4 - theta)
            verts[1, 3] = y_c + diagby2 * np.sin(7 * np.pi / 4 - theta)
        return np.matmul(verts, alpha)
