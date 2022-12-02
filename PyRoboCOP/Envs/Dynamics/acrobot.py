#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    The action is applying torque between two pendulum links.

    In this code. we describe the dynamics for acrobot in the general manipulator form given as the following:

    H(q)\\ddot{q} + C(q,\\dot{q})\\dot{q} +G(q) = B(q)u

    Using this dynamics description, we write the equations for the acceleration of the system and then
    discretize the dynamics.

    q=[\\theta_1,\\theta_2]
    Note that the \\theta_2 is relative to link 1

"""


import copy

import numpy as np
from Envs.Dynamics.param_dict_acrobot import param_dict
from numpy import cos, sin


class Acrobot(object):
    def __init__(self):

        self.param_dict = param_dict
        # param_dict contains all the parameters of the acrobot system
        self.m1 = self.param_dict["m1"]
        self.m2 = self.param_dict["m2"]
        self.L1 = self.param_dict["l1"]
        self.L2 = self.param_dict["l2"]
        self.lc1 = self.param_dict["lc1"]
        self.lc2 = self.param_dict["lc2"]
        self.I1 = self.param_dict["I1"]
        self.I2 = self.param_dict["I2"]
        self.dt = self.param_dict["dt"]
        self.g = self.param_dict["g"]

        # Next we list the dimension of the acrobot system
        # Note that these variables are expected by the OCP2NLP interface. if they are not provided, then it will throw an error

        self.n_d = 4
        self.n_a = 0
        self.n_cc = 0
        self.n_u = 1
        self.n_p = 0
        self.T = 100
        self.times = self.dt * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 0
        self.nnz_hess = 0

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
        # NEW
        return self.n_d, self.n_a, self.n_u, self.n_p, self.n_cc, self.T, self.times, self.nnz_jac, self.nnz_hess

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """

        x0 = np.array([0, 0, 0, 0])
        xdot0 = np.array([0, 0, 0, 0])
        u0 = np.array([0.0])
        y0 = np.array([0])

        return x0, xdot0, y0, u0

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """

        b_max = 40.0
        lb = np.array([-2 * np.pi, -2 * np.pi, -b_max, -b_max, -b_max, -b_max, -b_max, -b_max, -b_max])
        ub = np.array([2 * np.pi, 2 * np.pi, b_max, b_max, b_max, b_max, b_max, b_max, b_max])

        return lb, ub

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.array([0.0, 0.0, 0.0, 0.0])
        return xic

    def bounds_finaltime(self):
        """
        Method to return the bounds on final time state for the OCP instance
        """
        xfc = np.array([np.pi, 0, 0.0, 0.0])
        return xfc, xfc

    def objective(self, t, x, xdot, y, u, p):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        theta1 = x[0]
        theta2 = x[1]
        c = (theta1 - np.pi) ** 2 + (theta2) ** 2 + 0.1 * x[2] ** 2 + 0.1 * x[3] ** 2 + 0.1 * u[0] ** 2

        return c

    def constraint(self, c, t, x, xdot, y, u, p):
        """
        Method to return the constraint of the OCP instance
        c - return constraint residuals in this array
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """

        # The following equations describe the generalized manipulator dynamics for the acrobot system

        alpha = self.I1 + self.I2 + self.m2 * self.L1**2 + 2 * self.m2 * self.L1 * self.lc2 * cos(x[1])
        beta = self.I2 + self.m2 * self.L1 * self.lc2 * cos(x[1])
        gamma = self.I2 + self.m2 * self.L1 * self.lc2 * cos(x[1])
        delta = self.I2

        C11 = -2 * self.m2 * self.L1 * self.lc2 * sin(x[1]) * x[3]
        C12 = -self.m2 * self.L1 * self.lc2 * sin(x[1]) * x[3]
        C21 = self.m2 * self.L1 * self.lc2 * sin(x[1]) * x[2]
        C22 = 0

        g1 = (self.m1 * self.lc1 + self.m2 * self.L1) * self.g * sin(x[0]) + self.m2 * self.g * self.L2 * sin(
            x[0] + x[1]
        )
        g2 = self.m2 * self.g * self.L2 * sin(x[0] + x[1])

        M = np.array([[alpha, beta], [gamma, delta]])
        M_inv = np.array([[delta, -beta], [-gamma, alpha]]) / (alpha * delta - beta * gamma)
        C = np.array([[C11, C12], [C21, C22]])
        G = np.array([[g1], [g2]])

        qddot = np.array([[xdot[2]], [xdot[3]]])
        qdot = np.array([[x[2]], [x[3]]])

        # The next equation is the generalized manipulator dynamics

        c[0, 0] = -xdot[0] + x[2]
        c[1, 0] = -xdot[1] + x[3]
        n = -(np.matmul(C, qdot) + G)

        # Method 2: State Space formulation
        n[1] += u[0]
        try:
            c1 = np.matmul(M_inv, n)
            c[2, 0] = -xdot[2] + c1[0, 0]
            c[3, 0] = -xdot[3] + c1[1, 0]
        except:
            print("error")
