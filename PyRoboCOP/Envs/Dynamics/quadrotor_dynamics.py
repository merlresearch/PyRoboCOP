#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
    The state of the system is given by x=[x,y,z,\\psi, \\theta,\\phi,\\dot{x},\\dot{y},\\dot{z},p,q,r]
    This script describes the dynamical equations
    the state vector is 12 dimensional in the order written above

"""


import copy

import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t


class quadrotor_dyn(object):
    def __init__(self, params):
        self.params = params

    def get_dynamics_equations(self, x, u):

        psi = x[3]
        theta = x[4]
        phi = x[5]
        xdot = x[6]
        ydot = x[7]
        zdot = x[8]
        p = x[9]
        q = x[10]
        r = x[11]
        Ix = self.params["Ix"]
        Iy = self.params["Iy"]
        Iz = self.params["Iz"]

        dxdt = np.zeros_like(x)
        dxdt[0] = xdot
        dxdt[1] = ydot
        dxdt[2] = zdot
        dxdt[3] = q * s(phi) / c(theta) + r * c(phi) / c(theta)
        dxdt[4] = q * c(phi) - r * s(phi)
        dxdt[5] = p + q * (s(phi) * t(theta)) + r * (c(phi) * t(theta))
        dxdt[6] = -1.0 / self.params["m"] * (s(phi) * s(psi) + c(phi) * c(psi) * s(theta)) * u[0]
        dxdt[7] = -1.0 / self.params["m"] * (c(psi) * s(phi) - c(phi) * s(psi) * s(theta)) * u[0]
        dxdt[8] = self.params["g"] - 1.0 / self.params["m"] * (c(phi) * c(theta)) * u[0]
        dxdt[9] = (Iy - Iz) / Ix * q * r + 1.0 / Ix * u[1]
        dxdt[10] = (Iz - Ix) / Iy * p * r + 1.0 / Iy * u[2]
        dxdt[11] = (Ix - Iy) / Iz * p * q + 1.0 / Iz * u[3]

        return dxdt
