# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Class modeling collision avoidance between a controlled objects
"""


import adolc as ad
import numpy as np


class collisionAvoidance(object):
    """
    Polytope enclosing the controlled object is defined by the center and the vertices of polytope
    valgindex - starting index of the algebraic variables
    ocp - user-supplied OCP class
    objind1 - index of first object
    objdynamic1 - True if dynamic, False otherwise
    objnv1 - number of vertices
    objind2 - index of second object
    objdynamic2 - True if dynamic, False otherwise
    objnv2 - number of vertices
    """

    def __init__(self, valgindex, ocp, objind1, objdynamic1, objnv1, objind2, objdynamic2, objnv2, tol=1.0e-4):
        self.valgindex = valgindex
        self.ocp = ocp
        self.objind1 = objind1
        self.objdynamic1 = objdynamic1
        self.objnv1 = objnv1
        self.objind2 = objind2
        self.objdynamic2 = objdynamic2
        self.objnv2 = objnv2
        self.tol = tol

        self.nd = 3
        self.n_var = 2 * self.objnv1 + 2 * self.objnv2 + 3
        self.n_con = self.objnv1 + self.objnv2 + 3
        self.n_cc = self.objnv1 + self.objnv2

        self.cc_var1 = self.valgindex + np.arange(self.n_cc)
        self.cc_var2 = self.valgindex + self.n_cc + np.arange(self.n_cc)
        self.lbcon = np.zeros((self.n_con,))
        self.ubcon = np.zeros((self.n_con,))
        self.lb = np.zeros((self.n_var,))
        self.ub = np.Infinity * np.ones((self.n_var,))
        self.lb[-1] = -np.Infinity
        self.lb[-2] = -np.Infinity

    #    self.ub[-1] = 0.0
    #    self.ub[-2] = 0.0

    def constraint(self, x, xdot, y, u):
        con = np.zeros((self.n_con,))
        con = con.astype(object)

        alpha1 = y[self.valgindex : self.valgindex + self.objnv1]
        alpha2 = y[self.valgindex + self.objnv1 : self.valgindex + self.objnv1 + self.objnv2]
        nu1 = y[self.valgindex + self.objnv1 + self.objnv2 : self.valgindex + 2 * self.objnv1 + self.objnv2]
        nu2 = y[self.valgindex + 2 * self.objnv1 + self.objnv2 : self.valgindex + 2 * self.objnv1 + 2 * self.objnv2]
        slack = y[self.valgindex + 2 * self.objnv1 + 2 * self.objnv2]
        lam1 = y[self.valgindex + 2 * self.objnv1 + 2 * self.objnv2 + 1]
        lam2 = y[self.valgindex + 2 * self.objnv1 + 2 * self.objnv2 + 2]
        V1 = np.zeros((3, self.objnv1))
        V1 = V1.astype(object)
        self.ocp.get_object_vertices(V1, self.objind1, x, xdot, y[: self.ocp.n_a], u)
        v1 = np.matmul(V1, alpha1)
        V2 = np.zeros((3, self.objnv2))
        V2 = V2.astype(object)
        self.ocp.get_object_vertices(V2, self.objind2, x, xdot, y[: self.ocp.n_a], u)
        v2 = np.matmul(V2, alpha2)

        v1_v2 = v1 - v2
        con[0] = np.sqrt(np.dot(v1_v2.T, v1_v2) + self.tol) - np.sqrt(5 * self.tol) - slack
        ind = 1
        for i in range(self.objnv1):
            con[ind] = np.dot(V1[:, i], v1_v2) - nu1[i] + lam1
            ind = ind + 1
        for i in range(self.objnv2):
            con[ind] = -np.dot(V2[:, i], v1_v2) - nu2[i] + lam2
            ind = ind + 1
        con[ind] = np.sum(alpha1) - 1.0
        ind = ind + 1
        con[ind] = np.sum(alpha2) - 1.0
        ind = ind + 1
        return con
