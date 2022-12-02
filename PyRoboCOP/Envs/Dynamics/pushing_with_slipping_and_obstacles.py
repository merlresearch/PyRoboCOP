#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np
from Envs.Dynamics.planar_pushing_slipping import PlanarPusherSlipping


class pushingobstacles(object):
    def __init__(self):
        self.n_d = 4
        self.n_a = 4
        self.n_u = 2
        self.n_p = 0
        self.n_cc = 2
        self.T = 100
        self.dt = 0.1
        # self.times = (1./self.T)*np.array(list(range(self.T+1)))    # NEW
        self.times = self.dt * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 0
        self.nnz_hess = 0

        # self.dt = 0.5
        self.mu = 0.3

        self.pushing_dynamics = PlanarPusherSlipping(self.dt, self.mu)

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
        xlb = np.array([-5.0, -5.0, -2 * np.pi, -1])
        xub = np.array([5.0, 5.0, 2 * np.pi, 1])
        # Bounds for the state derivatives
        xdotlb = np.array([-5, -5, -5, -0.5])
        xdotub = np.array([5, 5, 5, 0.5])
        # Bounds of algebraic variable
        ylb = np.array([0.0, 0.0, 0, 0])
        yub = np.array([np.infty, np.infty, 0.5, 0.5])
        # Bounds of control inputs
        ulb = np.array([0, -1.0])
        uub = np.array([0.5, 1.0])

        # Stack them to get the ub and lb vector
        lb = np.hstack((xlb, xdotlb, ylb, ulb))
        ub = np.hstack((xub, xdotub, yub, uub))

        return lb, ub

    # def bounds_params(self):
    #     """
    #     Method to return the bounds on the parameters in the OCP instance
    #     lbp - lower bound array
    #     ubp - upper bound array
    #     """
    #     return np.array([0.01]), np.array([100.0])

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.array([0, 0, 0, 0])
        xdot0 = np.array([0, 0, 0, 0])
        y0 = np.array([0.0, 0.0, 0.0, 0.0])
        u0 = np.array([0, 0])
        return x0, xdot0, y0, u0

    # def initialpoint_params(self):
    #     """
    #     Method to return the initial guess for the parameters in the OCP instance

    #     """
    #     return np.array([10.0])

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = [0, 0, 0, 0.0]
        return xic

    def bounds_finaltime(self):
        """
        Method to return the bounds on the states at final time
        """
        lbxf = np.array([0.5, 0.5, 0.0, -0.5])
        ubxf = np.array([0.5, 0.5, 0.0, 0.5])
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
        c = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.0) ** 2 + (x[3]) ** 2
        return c

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
        ub[cc_var1[i]]-y[cc_var1[i]] >= 0 \perp y[cc_var2[i]]-lb[cc_var2[i]] >= 0
        if cc_bnd2 = 1, cc_bnd2 = 1
        ub[cc_var2[i]]-y[cc_var1[i]] >= 0 \perp ub[cc_var2[i]]-y[cc_var2[i]] >= 0
        """
        cc_var1 = np.reshape([1, 0], (self.n_cc, 1))
        cc_bnd1 = np.array([0, 0])
        cc_var2 = np.reshape([2, 3], (self.n_cc, 1))
        cc_bnd2 = np.array([0, 0])
        return cc_var1, cc_bnd1, cc_var2, cc_bnd2

    # def objective(self, t, x, xdot, y, u, params):
    #     """
    #     Method to return the objective function of the OCP instance
    #     x - numpy 1-d array of differential variables at time t
    #     xdot - numpy 1-d array of time derivative differential variables at time t
    #     y - numpy 1-d array of algebraic variables at time t
    #     u - numpy 1-d array of control avriables at time t
    #     params - parameters
    #     """
    #     return 0.0

    # def objective_mayer(self, x0, xf, params):
    #     """
    #     Method to return the objective function of the OCP instance that is not integral
    #     x - numpy 1-d array of differential variables at time t
    #     xdot - numpy 1-d array of time derivative differential variables at time t
    #     y - numpy 1-d array of algebraic variables at time t
    #     u - numpy 1-d array of control avriables at time t
    #     params - parameters
    #     """
    #     # DR: should we have instead only one obejctive function with a flag?
    #     tf = params[0]
    #     return tf

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
        tf = 1  # params[0]
        # use this relationship to replace input
        # u_p = y[2]-y[3]

        dxdt = self.pushing_dynamics.dynamics(x, u, y)
        c[0] = -xdot[0] + tf * dxdt[0]
        c[1] = -xdot[1] + tf * dxdt[1]
        c[2] = -xdot[2] + tf * dxdt[2]
        c[3] = -xdot[3] + tf * (y[2] - y[3])

        # # algebraic constraints for friction cone
        c[4] = y[0] - u[1] - self.mu * u[0]
        c[5] = y[1] + u[1] - self.mu * u[0]
        # return c
        # c[6] = u[2] - (y[2]-y[3])

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
        if ind == 0:
            # this is the obstacle
            nv = 4
            # verts = np.zeros((3, nv))
            corners = 0.05 * np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [0, 0, 0, 0]])
            verts[:, :] = corners + np.array([[0.5], [0.62], [0]])
            # verts[:, :] = corners + np.array([[0.30], [-0.1], [0]])
            # verts[:, :] = corners + np.array([[0.30], [0.4], [0]])

            # return verts
        if ind == 1:
            # this is the object - car
            nv = 4
            # verts = np.zeros((3, nv))
            # verts = verts.astype(object)
            sidexby2 = 0.045
            sideyby2 = 0.045
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
            # return verts
        if ind == 2:
            # this is an obstacle
            nv = 4
            # verts = np.zeros((3, nv))
            corners = 0.05 * np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [0, 0, 0, 0]])
            verts[:, :] = corners + np.array([[0.5], [0.38], [0]])
            # verts[:, :] = corners + np.array([[0.55], [-0.1], [0]])
            # verts[:, :] = corners + np.array([[0.55], [0.4], [0]])
            # return verts
