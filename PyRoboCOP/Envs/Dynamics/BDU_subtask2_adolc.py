#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""

Description: Class to the define the dynamics and constraints of the MPCC for subtask 2 of the Belt Drive Unit
"""


import copy
import pdb

import numpy as np


class loop_2_kps_OCP(object):
    def __init__(self, scenario=1):

        # Parameters required by solver
        self.n_d = 18  # pos + vel + ang + ang_vel: 6 + 6 + 3 + 3
        self.n_a = 3  # algebraic variable + obstacle avoidance: 2 +1
        self.n_u = 6  # x y z, ang*3
        self.n_p = 0
        self.n_cc = 1
        self.T = 50
        self.dt = 0.02
        self.times = self.dt * np.array(list(range(self.T + 1)))
        self.nnz_jac = 57
        self.nnz_hess = 0  # haven't used hessian

        # Parameters specific for the BDU system
        # 1, 2, 3, or 4 # See ICRA paper for explanation of the different scenarios
        self.scenario = scenario
        self.k1 = 1000  # position
        self.k2 = 1  # velocity
        self.k3 = 1e-5  # control input
        self.k5 = 100  # 100 #elastic force
        self.k6 = 100  # ang
        self.k7 = 1  # ang_vel
        self.k8 = 1e4

        if self.scenario == 1:
            try:
                self.init_pos = np.load("../Results/BDU/traj_subtask1_scenario123_x.npy")[-1, :6, 1]
            except:
                self.init_pos = np.array([0.55, 0.23, 0.5325, 0.55, 0.23, 0.34])
            # s1 target_pos is chosen based on second pulley center
            self.target_pos = np.array([0.39, 0.23, 0.35, 0.55, 0.23, 0.3])
            self.second_pulley_center = np.array([0.42, 0.23, 0.37])
        elif self.scenario == 2:
            try:
                self.init_pos = np.load("../Results/BDU/traj_subtask1_scenario123_x.npy")[-1, :6, 1]
            except:
                self.init_pos = np.array([0.55, 0.23, 0.5325, 0.55, 0.23, 0.34])
            self.target_pos = np.array([0.38 + 0.03807, 0.23, 0.38 + 0.0919, 0.55, 0.23, 0.3])  # s2
            self.second_pulley_center = np.array([0.42 + 0.03807, 0.23, 0.37 + 0.0919])
        elif self.scenario == 3:
            try:
                self.init_pos = np.load("../Results/BDU/traj_subtask1_scenario123_x.npy")[-1, :6, 1]
            except:
                self.init_pos = np.array([0.55, 0.23, 0.5325, 0.55, 0.23, 0.34])
            self.target_pos = np.array([0.45, 0.23, 0.255, 0.55, 0.23, 0.3])  # s3
            self.second_pulley_center = np.array([0.42 + 0.03484, 0.23, 0.37 - 0.065])
        elif self.scenario == 4:
            try:
                self.init_pos = np.load("../Results/BDU/traj_subtask1_scenario4_x.npy")[-1, :6, 1]
            except:
                self.init_pos = np.array([0.55, 0.23, 0.6325, 0.55, 0.23, 0.34])
            self.target_pos = np.array([0.285, 0.23, 0.35, 0.55, 0.23, 0.3])  # s4
            self.second_pulley_center = np.array([0.32, 0.23, 0.37])

        self.target_pos[0:3] += 0.00 * np.random.randn(3)  # 0.005
        self.target_vel = np.zeros(6)
        self.target_ang = np.array([0, -np.pi / 2, np.pi / 8 * 3])
        self.init_ang = np.zeros(3)

        self.loop_k = 63.34  # elastic coefficient
        self.damp_k = 4.613  # damping coefficient
        # -0.03 is because we use pulley center as bottom keypoint, radius of pulley 0.03
        self.length = 0.1418 - 0.03
        if self.scenario == 4:
            self.length = 0.2418 - 0.03
        self.pulley_center = np.array([0.55, 0.23, 0.37])
        self.mass_1 = 0.042
        self.mass_2 = 0.042
        self.target_u = np.zeros(self.n_u)
        # target elastic force
        self.target_y0 = (np.linalg.norm(self.target_pos[0:3] - self.pulley_center) - self.length) * self.loop_k

        # scaled obstacle-second pulley
        self.c1 = self.second_pulley_center[0] - 1
        self.c2 = self.second_pulley_center[1]
        self.c3 = self.second_pulley_center[2]
        self.r1 = 2
        self.r2 = 1
        self.r3 = 2

        # Complementarity info
        self.cc_var1 = np.array([0])
        self.cc_bnd1 = np.array([0])
        self.cc_var2 = np.array([1])
        self.cc_bnd2 = np.array([0])

    def get_info(self):
        """
        Method to return OCP info
        n_d - number of differential vars
        n_a - number of algebraic vars
        n_u - number or controls vars
        n_cc - number of complementarity variables (part of algebraic vars)
        T   - number of time-steps
        times - the time at start of each of the time intervals, an array of (T+1)
        nnz_jac - number of nonzeros in jacobian of DAE
        nnz_hess - number of nonzeros in hessian of OCP at each time-step
        """
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

        return self.cc_var1, self.cc_bnd1, self.cc_var2, self.cc_bnd2

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        x: pos, vel, ang, ang_vel 6+6+3+3
        """
        lb = np.hstack(
            (
                -np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7]),
                -0.27 * np.ones(6),
                -10 * np.ones(6),
                -1.0e30 * np.ones(self.n_d),
                np.array([self.target_y0 / 2, 0, 0]),
                -10 * np.ones(3),
                -np.array([0, 10, 200]),
            )
        )
        ub = np.hstack(
            (
                np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7]),
                0.27 * np.ones(6),
                10 * np.ones(6),
                1.0e30 * np.ones(self.n_d),
                1.0e30 * np.ones(self.n_a),
                10 * np.ones(3),
                np.array([0, 10, 200]),
            )
        )

        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """

        x0 = np.hstack((self.target_pos, np.zeros(6), self.target_ang, np.zeros(3)))
        xdot0 = np.zeros(self.n_d)
        u0 = np.zeros(self.n_u)
        y0 = np.zeros(self.n_a)

        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """

        # np.array([0.1, np.pi, 0., -.0]) #
        xic = np.hstack((self.init_pos, np.zeros(6), self.init_ang, np.zeros(3)))

        return xic

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        """
        pos = x[0:6]
        vel = x[6:12]
        ang = x[12:15]
        ang_vel = x[15:18]
        c = (
            self.k1 * np.linalg.norm(pos[0:3] - self.target_pos[0:3]) ** 2
            + self.k2 * np.linalg.norm(vel[0:3]) ** 2
            + self.k3 * np.linalg.norm(u - self.target_u) ** 2
            + self.k6 * np.linalg.norm(ang - self.target_ang) ** 2
            + self.k7 * np.linalg.norm(ang_vel) ** 2
            + self.k5 * np.linalg.norm(y[0] - self.target_y0) ** 2
            + self.k8 * (pos[1] - self.target_pos[1]) ** 2
        )

        return c

    def gradient(self, g, t, x, xdot, y, u, params):
        """
        Method to return the gradient of the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        """
        pos = np.expand_dims(x[0:3], axis=1)
        vel = np.expand_dims(x[6:9], axis=1)
        inputs = np.expand_dims(u - self.target_u, axis=1)
        target_pos = np.expand_dims(self.target_pos[0:3], axis=1)
        ang = np.expand_dims(x[12:15], axis=1)
        ang_vel = np.expand_dims(x[15:18], axis=1)
        target_ang = np.expand_dims(self.target_ang, axis=1)

        g[:, 0] = 0.0
        g[0:3] = self.k1 * 2 * (pos - target_pos)
        g[6:9] = self.k2 * 2 * (vel)
        g[12:15] = self.k6 * 2 * (ang - target_ang)
        g[15:18] = self.k7 * 2 * (ang_vel)
        g[2 * self.n_d] = self.k5 * 2 * (y[0] - self.target_y0)
        g[2 * self.n_d + self.n_a :] = self.k3 * 2 * inputs
        g[1] += self.k8 * 2 * (pos[1] - target_pos[1])

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        """
        # bottom keypoint does not move
        temp = np.hstack((np.zeros((6, 6)), np.eye(6)))
        A = np.zeros((12, 12)) + np.vstack((temp, np.zeros((6, 12))))
        B1 = np.vstack((np.zeros((6, 3)), np.eye(3), np.zeros((3, 3))))
        B2 = np.vstack((np.zeros((6, 3)), np.eye(3), np.zeros((3, 3))))
        B3 = np.zeros((12, 3))  # bottom keypoint does not move
        force = (x[0:3] - self.pulley_center) / np.linalg.norm(x[0:3] - self.pulley_center) * y[0]
        damping_force = (x[6:9]) * self.damp_k
        c[0:12, 0] = (
            -xdot[0:12]
            + np.matmul(A, x[0:12])
            + np.matmul(B1, u[0:3] / self.mass_1)
            + np.matmul(B2, -force / self.mass_1)
            + np.matmul(B3, force / self.mass_2)
            + np.matmul(B2, -damping_force / self.mass_1)
            + np.matmul(B3, damping_force / self.mass_2)
        )
        # gravity
        c[8] += -10
        c[11] += -0
        # ang
        c[12:15, 0] = -xdot[12:15] + x[15:18]
        c[15:18, 0] = -xdot[15:18] + u[3:6]

        c[-2] = -y[1] + y[0] / self.loop_k + self.length - np.linalg.norm(x[0:3] - self.pulley_center)
        c[-1] = -y[2] + self.distance(x)

    def distance(self, x):
        # ellipsoid obstacle avoidance
        dis = (
            (x[0] - self.c1) ** 2 / self.r1**2
            + (x[1] - self.c2) ** 2 / self.r2**2
            + (x[2] - self.c3) ** 2 / self.r3**2
            - 1e-4
        )

        return dis

    def jacobianstructure(self, row, col):
        """
        20-by-45 matrix
        row 0-17: x
        row 18: complementarity constraint
        row 19: obstacle avoidance
        col 0-35: x,xdot
        col 36-38: y0-y2
        col 39-44:u0-u5
        """

        row[:, 0] = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            6,
            6,
            6,
            7,
            7,
            7,
            8,
            8,
            8,
            6,
            7,
            8,
            6,
            7,
            8,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            18,
            18,
            18,
            18,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            19,
            19,
            19,
        ]

        col[:, 0] = [
            6,
            7,
            8,
            9,
            10,
            11,
            39,
            40,
            41,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            36,
            36,
            36,
            6,
            7,
            8,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            0,
            1,
            2,
            36,
            37,
            15,
            16,
            17,
            42,
            43,
            44,
            0,
            1,
            2,
            38,
        ]

    def jacobian(self, jac, t, x, xdot, y, u, params):
        """
        Method to return the jacobian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        """
        length = np.linalg.norm(x[0:3] - self.pulley_center)
        j11 = (1 / length - (x[0] - self.pulley_center[0]) ** 2 / (length**3)) * y[0]  # df0dx0
        j12 = -(x[0] - self.pulley_center[0]) * (x[1] - self.pulley_center[1]) / (length**3) * y[0]  # df0dx1
        j13 = -(x[0] - self.pulley_center[0]) * (x[2] - self.pulley_center[2]) / (length**3) * y[0]  # df0dx2
        j21 = -(x[1] - self.pulley_center[1]) * (x[0] - self.pulley_center[0]) / (length**3) * y[0]
        j22 = (1 / length - (x[1] - self.pulley_center[1]) ** 2 / (length**3)) * y[0]
        j23 = -(x[1] - self.pulley_center[1]) * (x[2] - self.pulley_center[2]) / (length**3) * y[0]
        j31 = -(x[2] - self.pulley_center[2]) * (x[0] - self.pulley_center[0]) / (length**3) * y[0]
        j32 = -(x[2] - self.pulley_center[2]) * (x[1] - self.pulley_center[1]) / (length**3) * y[0]
        j33 = (1 / length - (x[2] - self.pulley_center[2]) ** 2 / (length**3)) * y[0]
        # jacobian of elastic force
        j_force = np.array([j11, j12, j13, j21, j22, j23, j31, j32, j33])
        j7 = (x[0:3] - self.pulley_center) / length

        jac[:, 0] = np.hstack(
            (
                np.ones(6),
                np.ones(3) / self.mass_1,
                -j_force / self.mass_1,
                -j7 / self.mass_1,
                -np.ones(3) * (self.damp_k / self.mass_1),
                -np.ones(self.n_d),
                -j7,
                np.array([1 / self.loop_k, -1]),
                np.ones(6),
                np.array(
                    [
                        2 * (x[0] - self.c1) / self.r1**2,
                        2 * (x[1] - self.c2) / self.r2**2,
                        2 * (x[2] - self.c3) / self.r3**2,
                        -1,
                    ]
                ),
            )
        )
