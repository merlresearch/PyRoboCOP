#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""

Description: Class to the define the dynamics and constraints of the MPCC for subtask 1 of the Belt Drive Unit
"""


import numpy as np


class loop_2_kps_OCP(object):
    def __init__(self, scenario=123):

        # Parameters required by solver
        self.n_d = 12
        self.n_a = 5
        self.n_u = 3
        self.n_p = 0
        self.n_cc = 2
        self.dt = 0.03
        self.T = 200
        self.times = self.dt * np.array(list(range(self.T + 1)))
        self.nnz_jac = 92
        self.nnz_hess = 36

        # 123, or 4 See ICRA paper for explanation of the different scenarios
        self.scenario = scenario

        # Parameters specific for the BDU system
        self.k1 = 1000
        self.k2 = 10
        self.k3 = 1e-5
        self.k4 = 1
        self.k5 = 0  # 1e5

        if self.scenario == 123:
            self.target_pos = np.array([0.55, 0.23, 0.5325, 0.55, 0.23, 0.34])  # s123
            self.init_pos = np.array([0.55, 0.0, 0.4825, 0.55, 0.0, 0.34])  # s123
        elif self.scenario == 4:
            self.target_pos = np.array([0.55, 0.23, 0.6325, 0.55, 0.23, 0.34])  # s4
            self.init_pos = np.array([0.55, 0.0, 0.5825, 0.55, 0.0, 0.34])  # s4
        self.target_vel = np.zeros(6)
        self.target_pos[0:3] += 0.00 * np.random.randn(3)  # 0.005

        self.loop_k = 63.34  # 168.07 #stiffness coefficient
        self.damp_k = 4.613  # damping coefficient
        if self.scenario == 123:
            self.length = 0.1418  # loop original length
        if self.scenario == 4:
            self.length = 0.2418  # s4
        self.pulley_bottom = np.array([0.55, 0.23, 0.34])
        self.mass_1 = 0.042
        self.mass_2 = 0.042
        self.target_u = np.array(
            [0, 0, (self.target_pos[2] - self.target_pos[5] - self.length) * self.loop_k]
        )  # + np.array([1]))

        # obstacle-first pulley
        self.c1 = 0.55  # obstacle center
        self.c2 = 0.21
        self.c3 = 0.37
        self.r1 = 20  # 10 #obstacl radius
        self.r2 = 2
        self.r3 = 8

        # Complementarity info
        self.cc_var1 = np.reshape([0, 1], (2, 1))
        self.cc_bnd1 = np.array([0, 0])
        self.cc_var2 = np.reshape([2, 3], (2, 1))
        self.cc_bnd2 = np.array([0, 0])

        self.eps = 1e-8

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
        """
        # lb = np.hstack((np.array([-0.7]), -0.7*np.ones(5),-0.12*np.ones(self.n_d//2), -1.e30*np.ones(self.n_d), 0*np.ones(self.n_a) , -20*np.ones(self.n_u) ))
        lb = np.hstack(
            (
                np.array([-0.7]),
                -0.7 * np.ones(5),
                -0.12 * np.ones(self.n_d // 2),
                -1.0e30 * np.ones(self.n_d),
                np.array([0, 0, 0, self.eps, 0]),
                -20 * np.ones(self.n_u),
            )
        )
        ub = np.hstack(
            (
                np.array([0.7]),
                0.7 * np.ones(5),
                0.12 * np.ones(self.n_d // 2),
                1.0e30 * np.ones(self.n_d),
                1.0e30 * np.ones(self.n_a),
                20 * np.ones(self.n_u),
            )
        )

        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        if self.scenario == 123:
            x0 = np.hstack((np.array([0.55, 0.23, 0.4825]), np.array([0.55, 0.23, 0.34]), np.zeros(3), np.zeros(3)))
        elif self.scenario == 4:
            x0 = np.hstack((np.array([0.55, 0.23, 0.5825]), np.array([0.55, 0.23, 0.34]), np.zeros(3), np.zeros(3)))
        xdot0 = np.zeros(self.n_d)
        u0 = np.zeros(self.n_u)
        y0 = np.zeros(self.n_a)

        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.hstack((self.init_pos, np.zeros(3), np.zeros(3)))  # np.array([0.1, np.pi, 0., -.0]) #

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
        pos = x[0 : self.n_d // 2]
        vel = x[self.n_d // 2 :]
        c = (
            self.k1 * np.linalg.norm(pos - self.target_pos) ** 2
            + self.k2 * np.linalg.norm(vel) ** 2
            + self.k3 * np.linalg.norm(u - self.target_u) ** 2
            + self.k5 * (pos[1] - self.target_pos[1]) ** 2
            + self.k5 * (pos[4] - self.target_pos[4]) ** 2
        )

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
        g[:, 0] = 0.0
        pos = np.expand_dims(x[0 : self.n_d // 2], axis=1)
        vel = np.expand_dims(x[self.n_d // 2 :], axis=1)
        inputs = np.expand_dims(u - self.target_u, axis=1)
        target_pos = np.expand_dims(self.target_pos, axis=1)
        temp1 = x[1] - self.target_pos[1]
        temp2 = x[4] - self.target_pos[4]

        g[0 : self.n_d // 2] = self.k1 * 2 * (pos - target_pos)
        g[self.n_d // 2 : self.n_d] = self.k2 * 2 * (vel)
        g[2 * self.n_d + self.n_a :] = self.k3 * 2 * inputs

        g[1] += self.k5 * 2 * temp1
        g[4] += self.k5 * 2 * temp2

    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """

        temp = np.hstack((np.zeros((self.n_d // 2, self.n_d // 2)), np.eye(self.n_d // 2)))
        A = np.zeros((self.n_d, self.n_d)) + np.vstack((temp, np.zeros((self.n_d // 2, self.n_d))))
        B1 = np.vstack((np.zeros((self.n_d // 2, self.n_u)), np.eye(self.n_u), np.zeros((self.n_d // 4, self.n_u))))
        B2 = np.vstack((np.zeros((self.n_d // 2, self.n_u)), np.eye(self.n_u), np.zeros((self.n_d // 4, self.n_u))))
        B3 = np.vstack((np.zeros((self.n_d // 2, self.n_u)), np.zeros((self.n_d // 4, self.n_u)), np.eye(self.n_u)))
        B4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # y[0]: elastic force, y[1]:normal force
        force = (x[0:3] - x[3:6]) / np.linalg.norm(x[0:3] - x[3:6]) * y[0]
        damping_force = (x[6:9] - x[9:12]) * self.damp_k
        c[0:12, 0] = (
            -xdot
            + np.matmul(A, x)
            + np.matmul(B1, u / self.mass_1)
            + np.matmul(B2, -force / self.mass_1)
            + np.matmul(B3, force / self.mass_2)
            + np.matmul(B2, -damping_force / self.mass_1)
            + np.matmul(B3, damping_force / self.mass_2)
            + B4 * (-y[1]) / self.mass_2
        )
        # gravity
        c[8] += -10
        c[11] += -10
        c[-3] = -y[2] + y[0] / self.loop_k + self.length - np.linalg.norm(x[0:3] - x[3:6])
        c[-2] = -y[3] + self.k4 * (np.linalg.norm(x[3:6] - self.pulley_bottom) ** 2 + self.eps**2) ** 0.5
        c[-1] = -y[4] + self.distance(x)

    def distance(self, x):
        # ellipsoid obstacle avoidance
        # distance is scaled on self.r1, r2, r3 and 1e-4
        dis = (
            (x[3] - self.c1) ** 2 / self.r1**2
            + (x[4] - self.c2) ** 2 / self.r2**2
            + (x[5] - self.c3) ** 2 / self.r3**2
            - 1e-4
        )

        return dis

    def jacobianstructure(self, row, col):

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
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            8,
            6,
            7,
            8,
            6,
            7,
            8,
            6,
            7,
            8,
            9,
            9,
            9,
            9,
            9,
            9,
            10,
            10,
            10,
            10,
            10,
            10,
            11,
            11,
            11,
            11,
            11,
            11,
            9,
            10,
            11,
            9,
            10,
            11,
            9,
            10,
            11,
            11,
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
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            13,
            13,
            13,
            13,
            14,
            14,
            14,
            14,
        ]

        col[:, 0] = [
            6,
            7,
            8,
            9,
            10,
            11,
            29,
            30,
            31,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            24,
            24,
            24,
            6,
            7,
            8,
            9,
            10,
            11,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            24,
            24,
            24,
            6,
            7,
            8,
            9,
            10,
            11,
            25,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            0,
            1,
            2,
            3,
            4,
            5,
            24,
            26,
            3,
            4,
            5,
            27,
            3,
            4,
            5,
            28,
        ]

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
        length = np.linalg.norm(x[0:3] - x[3:6])
        # jacobian of the elastic force
        j11 = (1 / length - (x[0] - x[3]) ** 2 / (length**3)) * y[0]  # df0dx0
        j12 = -(x[0] - x[3]) * (x[1] - x[4]) / (length**3) * y[0]  # df0dx1
        j13 = -(x[0] - x[3]) * (x[2] - x[5]) / (length**3) * y[0]  # df0dx2
        j14 = -j11  # df0dx3
        j15 = -j12  # df0dx4
        j16 = -j13  # df0dx5
        j21 = -(x[1] - x[4]) * (x[0] - x[3]) / (length**3) * y[0]  # df1dx0
        j22 = (1 / length - (x[1] - x[4]) ** 2 / (length**3)) * y[0]
        j23 = -(x[1] - x[4]) * (x[2] - x[5]) / (length**3) * y[0]
        j24 = -j21
        j25 = -j22
        j26 = -j23
        j31 = -(x[2] - x[5]) * (x[0] - x[3]) / (length**3) * y[0]
        j32 = -(x[2] - x[5]) * (x[1] - x[4]) / (length**3) * y[0]
        j33 = (1 / length - (x[2] - x[5]) ** 2 / (length**3)) * y[0]
        j34 = -j31
        j35 = -j32
        j36 = -j33
        j_force = np.array([j11, j12, j13, j14, j15, j16, j21, j22, j23, j24, j25, j26, j31, j32, j33, j34, j35, j36])
        # jacobian of the length
        j7 = (x[0:3] - x[3:6]) / length
        # distance to the pulley bottom
        # j8 = self.k4*2*(x[3:6] - self.pulley_bottom)
        j8 = (
            self.k4
            * (x[3:6] - self.pulley_bottom)
            / (np.linalg.norm(x[3:6] - self.pulley_bottom) ** 2 + self.eps**2) ** 0.5
        )

        jac[:, 0] = np.hstack(
            (
                np.ones(self.n_d // 2),
                np.ones(self.n_u) / self.mass_1,
                -j_force / self.mass_1,
                -j7 / self.mass_1,
                -np.ones(3) * (self.damp_k / self.mass_1),
                np.ones(3) * (self.damp_k / self.mass_1),
                j_force / self.mass_1,
                j7 / self.mass_1,
                np.ones(3) * (self.damp_k / self.mass_2),
                -np.ones(3) * (self.damp_k / self.mass_2),
                np.array([-1 / self.mass_2]),
                -np.ones(self.n_d),
                -j7,
                j7,
                np.array([1 / self.loop_k, -1]),
                j8,
                np.array([-1]),
                np.array(
                    [
                        2 * (x[3] - self.c1) / self.r1**2,
                        2 * (x[4] - self.c2) / self.r2**2,
                        2 * (x[5] - self.c3) / self.r3**2,
                        -1,
                    ]
                ),
            )
        )

    def hessianstructure(self, row, col):

        # hessian structure only use lower triangular matrix
        row[:, 0] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                2,
                3,
                4,
                5,
                3,
                4,
                5,
                4,
                5,
                5,
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
                29,
                30,
                31,
            ]
        )
        col[:, 0] = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                5,
                24,
                24,
                24,
                24,
                24,
                24,
                6,
                7,
                8,
                9,
                10,
                11,
                29,
                30,
                31,
            ]
        )

        return row, col

    def hessian(self, hesst, t, x, xdot, y, u, params, mult, obj_factor):
        """
        Method to return the hessian of the Lagrangian of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        # hesst = np.zeros((self.nnz_hess,1))

        length = np.linalg.norm(x[0:3] - x[3:6])
        dlengthdx0 = (x[0] - x[3]) / length
        dlengthdx1 = (x[1] - x[4]) / length
        dlengthdx2 = (x[2] - x[5]) / length
        dlengthdx3 = -dlengthdx0
        dlengthdx4 = -dlengthdx1
        dlengthdx5 = -dlengthdx2

        # hessian of complementarity constraint
        j11 = 1 / length - (x[0] - x[3]) ** 2 / (length**3)  # df0dx0
        j12 = -(x[0] - x[3]) * (x[1] - x[4]) / (length**3)  # df0dx1
        j13 = -(x[0] - x[3]) * (x[2] - x[5]) / (length**3)  # df0dx2
        j14 = -j11
        j15 = -j12
        j16 = -j13
        j21 = -(x[1] - x[4]) * (x[0] - x[3]) / (length**3)
        j22 = 1 / length - (x[1] - x[4]) ** 2 / (length**3)
        j23 = -(x[1] - x[4]) * (x[2] - x[5]) / (length**3)
        j24 = -j21
        j25 = -j22
        j26 = -j23
        j31 = -(x[2] - x[5]) * (x[0] - x[3]) / (length**3)
        j32 = -(x[2] - x[5]) * (x[1] - x[4]) / (length**3)
        j33 = 1 / length - (x[2] - x[5]) ** 2 / (length**3)
        j34 = -j31
        j35 = -j32
        j36 = -j33
        j_force = np.array(
            [[j11, j21, j31], [j12, j22, j32], [j13, j23, j33], [j14, j24, j34], [j15, j25, j35], [j16, j26, j36]]
        )
        hc1 = -np.hstack((j_force, -j_force))

        # hessian of elastic force
        h111 = (
            -dlengthdx0 / length**2
            - (2 * (x[0] - x[3]) * length - 3 * (x[0] - x[3]) ** 2 * dlengthdx0) / length**4
        ) * y[0]
        h112 = (-dlengthdx1 / length**2 - (-3 * (x[0] - x[3]) ** 2 * dlengthdx1) / length**4) * y[0]
        h113 = (-dlengthdx2 / length**2 - (-3 * (x[0] - x[3]) ** 2 * dlengthdx2) / length**4) * y[0]
        h114 = -h111
        h115 = -h112
        h116 = -h113

        h121 = (-(x[1] - x[4]) / length**3 + 3 * (x[0] - x[3]) * (x[1] - x[4]) * dlengthdx0 / length**4) * y[0]
        h122 = (-(x[0] - x[3]) / length**3 + 3 * (x[0] - x[3]) * (x[1] - x[4]) * dlengthdx1 / length**4) * y[0]
        h123 = (+3 * (x[0] - x[3]) * (x[1] - x[4]) * dlengthdx2 / length**4) * y[0]
        h124 = -h121
        h125 = -h122
        h126 = -h123

        h131 = (-(x[2] - x[5]) / length**3 + 3 * (x[0] - x[3]) * (x[2] - x[5]) * dlengthdx0 / length**4) * y[0]
        h132 = (+3 * (x[0] - x[3]) * (x[2] - x[5]) * dlengthdx1 / length**4) * y[0]
        h133 = (-(x[0] - x[3]) / length**3 + 3 * (x[0] - x[3]) * (x[2] - x[5]) * dlengthdx2 / length**4) * y[0]
        h134 = -h131
        h135 = -h132
        h136 = -h133

        h1_left = np.array(
            [
                [h111, h121, h131],
                [h112, h122, h132],
                [h113, h123, h133],
                [h114, h124, h134],
                [h115, h125, h135],
                [h116, h126, h136],
            ]
        )
        h1_right = -h1_left
        h1 = np.hstack((h1_left, h1_right))

        h211 = h121
        h212 = h122
        h213 = h123
        h214 = h124
        h215 = h125
        h216 = h126

        h221 = (-dlengthdx0 / length**2 - (-3 * (x[1] - x[4]) ** 2 * dlengthdx0) / length**4) * y[0]
        h222 = (
            -dlengthdx1 / length**2
            - (2 * (x[1] - x[4]) * length - 3 * (x[1] - x[4]) ** 2 * dlengthdx1) / length**4
        ) * y[0]
        h223 = (-dlengthdx2 / length**2 - (-3 * (x[1] - x[4]) ** 2 * dlengthdx2) / length**4) * y[0]
        h224 = -h221
        h225 = -h222
        h226 = -h223

        h231 = (+3 * (x[1] - x[4]) * (x[2] - x[5]) * dlengthdx0 / length**4) * y[0]
        h232 = (-(x[2] - x[5]) / length**3 + 3 * (x[1] - x[4]) * (x[2] - x[5]) * dlengthdx1 / length**4) * y[0]
        h233 = (-(x[1] - x[4]) / length**3 + 3 * (x[1] - x[4]) * (x[2] - x[5]) * dlengthdx2 / length**4) * y[0]
        h234 = -h231
        h235 = -h232
        h236 = -h233

        h2_left = np.array(
            [
                [h211, h221, h231],
                [h212, h222, h232],
                [h213, h223, h233],
                [h214, h224, h234],
                [h215, h225, h235],
                [h216, h226, h236],
            ]
        )
        h2_right = -h2_left
        h2 = np.hstack((h2_left, h2_right))

        h311 = h131
        h312 = h132
        h313 = h133
        h314 = h134
        h315 = h135
        h316 = h136

        h321 = h231
        h322 = h232
        h323 = h233
        h324 = h234
        h325 = h235
        h326 = h236

        h331 = (-dlengthdx0 / length**2 - (-3 * (x[2] - x[5]) ** 2 * dlengthdx0) / length**4) * y[0]
        h332 = (-dlengthdx1 / length**2 - (-3 * (x[2] - x[5]) ** 2 * dlengthdx1) / length**4) * y[0]
        h333 = (
            -dlengthdx2 / length**2
            - (2 * (x[2] - x[5]) * length - 3 * (x[2] - x[5]) ** 2 * dlengthdx2) / length**4
        ) * y[0]
        h334 = -h331
        h335 = -h332
        h336 = -h333

        h3_left = np.array(
            [
                [h311, h321, h331],
                [h312, h322, h332],
                [h313, h323, h333],
                [h314, h324, h334],
                [h315, h325, h335],
                [h316, h326, h336],
            ]
        )
        h3_right = -h3_left
        h3 = np.hstack((h3_left, h3_right))

        h11y = 1 / length - (x[0] - x[3]) ** 2 / (length**3)
        h12y = -(x[0] - x[3]) * (x[1] - x[4]) / (length**3)
        h13y = -(x[0] - x[3]) * (x[2] - x[5]) / (length**3)
        h14y = -h11y
        h15y = -h12y
        h16y = -h13y
        h21y = -(x[1] - x[4]) * (x[0] - x[3]) / (length**3)
        h22y = 1 / length - (x[1] - x[4]) ** 2 / (length**3)
        h23y = -(x[1] - x[4]) * (x[2] - x[5]) / (length**3)
        h24y = -h21y
        h25y = -h22y
        h26y = -h23y
        h31y = -(x[2] - x[5]) * (x[0] - x[3]) / (length**3)
        h32y = -(x[2] - x[5]) * (x[1] - x[4]) / (length**3)
        h33y = 1 / length - (x[2] - x[5]) ** 2 / (length**3)
        h34y = -h31y
        h35y = -h32y
        h36y = -h33y

        hess_matrix = np.zeros((self.n_d * 2 + self.n_a + self.n_u, self.n_d * 2 + self.n_a + self.n_u))
        # hessian of elastic force
        hess_matrix[0:6, 0:6] += (
            mult[6] * (-h1 / self.mass_1)
            + mult[7] * (-h2 / self.mass_1)
            + mult[8] * (-h3 / self.mass_1)
            + mult[9] * (h1 / self.mass_2)
            + mult[10] * (h2 / self.mass_2)
            + mult[11] * (h3 / self.mass_2)
        )
        hess_matrix[0:6, 24] += (
            mult[6] * (-np.array([h11y, h12y, h13y, h14y, h15y, h16y]) / self.mass_1)
            + mult[7] * (-np.array([h21y, h22y, h23y, h24y, h25y, h26y]) / self.mass_1)
            + mult[8] * (-np.array([h31y, h32y, h33y, h34y, h35y, h36y]) / self.mass_1)
            + mult[9] * (np.array([h11y, h12y, h13y, h14y, h15y, h16y]) / self.mass_2)
            + mult[10] * (np.array([h21y, h22y, h23y, h24y, h25y, h26y]) / self.mass_2)
            + mult[11] * (np.array([h31y, h32y, h33y, h34y, h35y, h36y]) / self.mass_2)
        )
        hess_matrix[24, 0:6] += (
            mult[6] * (-np.array([h11y, h12y, h13y, h14y, h15y, h16y]) / self.mass_1)
            + mult[7] * (-np.array([h21y, h22y, h23y, h24y, h25y, h26y]) / self.mass_1)
            + mult[8] * (-np.array([h31y, h32y, h33y, h34y, h35y, h36y]) / self.mass_1)
            + mult[9] * (np.array([h11y, h12y, h13y, h14y, h15y, h16y]) / self.mass_2)
            + mult[10] * (np.array([h21y, h22y, h23y, h24y, h25y, h26y]) / self.mass_2)
            + mult[11] * (np.array([h31y, h32y, h33y, h34y, h35y, h36y]) / self.mass_2)
        )

        # hessian of objective function
        hess_matrix[0:6, 0:6] += self.k1 * 2 * obj_factor * np.eye(6)
        hess_matrix[6:12, 6:12] += self.k2 * 2 * obj_factor * np.eye(6)
        hess_matrix[29:, 29:] += self.k3 * 2 * obj_factor * np.eye(3)
        hess_matrix[1, 1] += self.k5 * 2 * obj_factor
        hess_matrix[4, 4] += self.k5 * 2 * obj_factor

        # hessian of obstacle avoidance
        hess_matrix[3, 3] += mult[14] * 2 / self.r1**2
        hess_matrix[4, 4] += mult[14] * 2 / self.r2**2
        hess_matrix[5, 5] += mult[14] * 2 / self.r3**2

        # complementarity constraint
        hess_matrix[0:6, 0:6] += mult[12] * hc1
        hess_matrix[3, 3] += (
            mult[13] * self.k4 * self.eps**2 / ((x[3] - self.pulley_bottom[0]) ** 2 + self.eps**2) ** 1.5
        )
        hess_matrix[4, 4] += (
            mult[13] * self.k4 * self.eps**2 / ((x[4] - self.pulley_bottom[1]) ** 2 + self.eps**2) ** 1.5
        )
        hess_matrix[5, 5] += (
            mult[13] * self.k4 * self.eps**2 / ((x[5] - self.pulley_bottom[2]) ** 2 + self.eps**2) ** 1.5
        )

        r = np.zeros((self.nnz_hess, 1), dtype=int)
        c = np.zeros((self.nnz_hess, 1), dtype=int)
        self.hessianstructure(r, c)
        # r,c = self.hessianstructure()
        for i in range(self.nnz_hess):
            hesst[i, 0] = hess_matrix[r[i, 0], c[i, 0]]

        # return hesst
