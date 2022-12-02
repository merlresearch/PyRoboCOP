#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nprnd
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import integrate


class PlanarPusherSlipping(object):
    def __init__(self, dt, mu):

        # coefficient of friction
        self.mu = mu

        # time discretization
        self.dt = dt

        # gravity
        self.g = 9.81

        # half length of object
        self.hl = 0.045

        # intgrate to mode_ind max moment coeff
        def func(x, y):
            return np.sqrt(x**2 + y**2)

        m = integrate.dblquad(func, -self.hl, self.hl, -self.hl, self.hl)
        mcoeff = m[0] / (2 * self.hl) ** 2

        # LS: 0.5 * w_f' * L * w_f <= 1
        self.L = (2.0 / (0.5 * self.g)) ** 2.0 * np.diagflat(np.array([1.0, 1.0, (1.0 / mcoeff) ** 2]))

        # face_index [left, top, right, bottom] (not used in UnitPusher)
        self.hybrid_modes = np.array([0, 1, 2, 3])

        # pusher positions in object frame
        # [left face, top face, right face, bottom face]
        self.p_x = np.array([-1.0, 0.0, 1.0, 0.0]) * self.hl
        self.p_y = np.array([0.0, 1.0, 0.0, -1.0]) * self.hl

        # pusher radius
        self.rpush = 0.005  # 5 mm

        # Jacobians from contact to object frame [2, 3, 4].
        # Different faces along third dimension

    def dynamics(self, x_k, u_k, y_k):
        """discrete dynamics: x_{k+1} = f(x_k, u_k)"""
        """ deriv_opt corresponds to the value : C^0 to C^2"""

        # pusher twist in object frame

        u_p = y_k[2] - y_k[3]
        self.p_y = (x_k[-1] + self.dt * u_p) * self.hl

        self.J_p = np.array(
            [[1, 0, -self.p_y], [0, 1, self.p_x[0]]],
        )
        w_p = np.zeros((3,))

        w_p = np.dot(self.J_p.T, u_k[0:2])
        # quasi-static assumpiton (w_f = -w_p)
        w_f = -1.0 * w_p

        # object twist in object frame v = -L*w_f
        vb = -1.0 * np.dot(self.L, w_f)

        # object twist in world frame
        R = np.array([[np.cos(x_k[2]), -1.0 * np.sin(x_k[2]), 0], [np.sin(x_k[2]), np.cos(x_k[2]), 0], [0, 0, 1]])

        self.dt * np.dot(R, vb)

        return np.dot(R, vb)

    def corners(self):
        """returns corners of slider as (2, 5) numpy array"""

        return np.array(
            [[-self.hl, self.hl, self.hl, -self.hl, -self.hl], [self.hl, self.hl, -self.hl, -self.hl, self.hl]]
        )
