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
        self.T = 60
        self.times = 0.1 * np.array(list(range(self.T + 1)))
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
        xic = np.array([1.0, 4.0, 0.0, 0.0])
        return xic

    def bounds_finaltime(self):
        """
        Method to return the bounds on final time state for the OCP instance
        """
        xfc = np.array([2.0, 2.5, np.pi / 2, 0.0])
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
        c = c + (x[1] - 2.5) * (x[1] - 2.5)
        c = c + (x[2] - np.pi / 3) * (x[2] - np.pi / 3)
        c = c + x[3] * x[3]
        c = c + u[0] * u[0]
        c = c + u[1] * u[1]
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
        c[0] = -xdot[0] + x[3] * np.sin(x[2])
        c[1] = -xdot[1] + x[3] * np.cos(x[2])
        c[2] = -xdot[2] + u[0] * x[3]
        c[3] = -xdot[3] + u[1]

    def get_objects_info(self):
        """
        Method to return the info on the objects (assumed to be polytopic)
        n_objects - number of objects
        object_dynamic - a boolean array set to True if the object is dynamic, False otherwise
        n_vertices - array indicating the number of vertices in the polytope bounding the objects
        """
        n_objects = 3
        object_dynamic = [False, True, False]
        n_vertices = [4, 4, 4]
        return n_objects, object_dynamic, n_vertices

    def get_object_vertices(self, verts, ind, x, xdot, y, u):
        """
        The vertices of the polytope (in 3D) bounding the object are assumed to be of the form
        V(x,xdot,y,u) = R(x,x,dot,y) * V_0 + q_c(x,xdot,y,u)
        where R(x,xdot,y,u) is the rotation matrix, V_0 is matrix with the vertices of the polytope in the
        columns (when objective is positioned at origin), q_c is the position of the object.
        V, V0 are 3 x (# vertices) matrices
        """
        # y_obst =
        if ind == 0:
            # this is the obstacle
            nv = 4
            corners = 0.5 * np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [0, 0, 0, 0]])
            verts[:, :] = corners + np.array([[2.0], [1.65], [0]])

        if ind == 1:
            # this is the object - car
            nv = 4
            sidexby2 = 0.25
            sideyby2 = 0.25
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

        if ind == 2:
            # this is an obstacle
            nv = 4
            corners = 0.5 * np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [0, 0, 0, 0]])
            verts[:, :] = corners + np.array([[2.0], [3.35], [0]])
