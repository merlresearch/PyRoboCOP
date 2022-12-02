#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from inspect import ismethod

import numpy as np
import Solvers.ocp2nlp.collisionAvoidance as ca
import Solvers.ocp2nlp.staticObstacle as so

# Checks is a class implements method


def method_exists(instance, method):
    return hasattr(instance, method) and ismethod(getattr(instance, method))


class CollisionAvoidanceOCP(object):
    def __init__(self, ocp):

        self.ocp = ocp

        (
            self.n_d,
            self.n_a,
            self.n_u,
            self.n_p,
            self.n_cc,
            self.T,
            self.times,
            self.nnz_jac,
            self.nnz_hess,
        ) = self.ocp.get_info()

        if self.n_cc > 0:
            self.cc_var1, self.cc_bnd1, self.cc_var2, self.cc_bnd2 = self.ocp.get_complementarity_info()

        # get the number of obstacles and controlled objects
        n_a = self.n_a
        n_cc = self.n_cc
        self.n_objects, self.objects_dynamic, self.objects_n_vertices = self.ocp.get_objects_info()
        self.n_collision_avoidance_cons = 0
        # get all of the collision avoidance constraints
        ind = 0
        self.collision_avoidance_cons = []
        for io1 in range(self.n_objects):
            for io2 in range(io1 + 1, self.n_objects):
                if self.objects_dynamic[io1] or self.objects_dynamic[io2]:
                    self.collision_avoidance_cons.append(
                        ca.collisionAvoidance(
                            n_a,
                            ocp,
                            io1,
                            self.objects_dynamic[io1],
                            self.objects_n_vertices[io1],
                            io2,
                            self.objects_dynamic[io2],
                            self.objects_n_vertices[io2],
                        )
                    )
                    n_a = n_a + self.collision_avoidance_cons[ind].n_var
                    n_cc = n_cc + self.collision_avoidance_cons[ind].n_cc
                    ind = ind + 1
        self.n_collision_avoidance_cons = ind

        self.n_d_new = self.n_d
        self.n_a_new = n_a
        self.n_u_new = self.n_u
        self.n_cc_new = n_cc

        # complementarity constraints
        self.cc_bnd1_new = np.zeros((self.n_cc_new,))
        self.cc_bnd2_new = np.zeros((self.n_cc_new,))
        if self.n_cc > 0:
            self.cc_bnd1_new[: self.n_cc] = np.copy(self.cc_bnd1)
            self.cc_bnd2_new[: self.n_cc] = np.copy(self.cc_bnd2)

        self.cc_var1_new = np.zeros((self.n_cc_new,), dtype=int)
        self.cc_var2_new = np.zeros((self.n_cc_new,), dtype=int)
        if self.n_cc > 0:
            self.cc_var1_new[: self.n_cc] = np.copy(self.cc_var1[:, 0])
            self.cc_var2_new[: self.n_cc] = np.copy(self.cc_var2[:, 0])

        ind = self.n_cc
        for ic in range(self.n_collision_avoidance_cons):
            for k in range(self.collision_avoidance_cons[ic].n_cc):
                self.cc_var1_new[ind] = self.collision_avoidance_cons[ic].cc_var1[k]
                self.cc_var2_new[ind] = self.collision_avoidance_cons[ic].cc_var2[k]
                ind = ind + 1

    def get_info(self):
        """
        Method to return OCP info
        n_d - number of differential vars
        n_a - number of algebraic vars
        n_u - number of controls vars
        n_p - number of parameters
        n_cc - number of complementarity variables (part of algebraic vars)
        T   - number of time-steps
        times - the time at start of each of the time intervals, an array of (T+1)
        nnz_jac - number of nonzeros in jacobian of DAE
        nnz_hess - number of nonzeros in hessian of OCP at each time-step
        """
        # NEW
        return (
            self.n_d_new,
            self.n_a_new,
            self.n_u_new,
            self.n_p,
            self.n_cc_new,
            self.T,
            self.times,
            self.nnz_jac,
            self.nnz_hess,
        )

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
        return self.cc_var1_new, self.cc_bnd1_new, self.cc_var2_new, self.cc_bnd2_new

    def bounds(self, t):
        lb = np.zeros((2 * self.n_d_new + self.n_a_new + self.n_u_new,))
        ub = np.zeros((2 * self.n_d_new + self.n_a_new + self.n_u_new,))
        lb_orig, ub_orig = self.ocp.bounds(t)
        lb[: 2 * self.n_d + self.n_a] = np.copy(lb_orig[: 2 * self.n_d + self.n_a])
        ub[: 2 * self.n_d + self.n_a] = np.copy(ub_orig[: 2 * self.n_d + self.n_a])
        ind = 2 * self.n_d + self.n_a
        for ic in range(self.n_collision_avoidance_cons):
            for k in range(self.collision_avoidance_cons[ic].n_var):
                lb[ind] = self.collision_avoidance_cons[ic].lb[k]
                ub[ind] = self.collision_avoidance_cons[ic].ub[k]
                ind = ind + 1
        lb[ind:] = lb_orig[2 * self.n_d + self.n_a :]
        ub[ind:] = ub_orig[2 * self.n_d + self.n_a :]
        return lb, ub

    def bounds_params(self):
        if self.n_p > 0:
            return self.ocp.bounds_params()

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.zeros((self.n_d_new,))
        xdot0 = np.zeros((self.n_d_new,))
        y0 = 0.5 * np.zeros((self.n_a_new,))
        u0 = np.zeros((self.n_u_new,))
        x0, xdot0, y0_orig, u0 = self.ocp.initialpoint(t)
        y0[: self.n_a] = np.copy(y0_orig)
        return x0, xdot0, y0, u0

    def initialpoint_params(self):
        if self.n_p > 0:
            return self.ocp.initialpoint_params()

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        return self.ocp.initialcondition()

    def bounds_finaltime(self):
        if method_exists(self.ocp, "bounds_finaltime"):
            return self.ocp.bounds_finaltime()

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control variables at time t
        params - parameters
        """
        return self.ocp.objective(t, x, xdot, y[: self.n_a], u, params)

    def objective_mayer(self, x0, xf, params):
        """
        Method to return the non-integral objective function of the OCP instance
        x0 - state at initial time
        xf - state at final time
        params - parameters
        """
        if method_exists(self.ocp, "objective_mayer"):
            return self.ocp.objective_mayer(x0, xf, params)
        else:
            return 0.0

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control variables at time t
        params - parameters
        """
        self.ocp.constraint(c[: self.n_d + self.n_a - self.n_cc], t, x, xdot, y[: self.n_a], u, params)
        ind = self.n_d + self.n_a - self.n_cc
        for ic in range(self.n_collision_avoidance_cons):
            c[ind : ind + self.collision_avoidance_cons[ic].n_con, 0] = self.collision_avoidance_cons[ic].constraint(
                x, xdot, y, u
            )
            ind = ind + self.collision_avoidance_cons[ic].n_con
