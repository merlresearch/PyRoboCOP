# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np


def simplexProjection(x):
    """
    Project x (nx x 1) numpy array onto the simplex.
    Weiran Wang Miguel A. Carreira-Perpinan,
    Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application,
    https://arxiv.org/pdf/1309.1541.pdf
    """
    nx = np.shape(x)[0]
    u = np.sort(x, axis=None)[::-1]  # flatten the array before sorting
    xx = [j * np.sign(u[j] + 1 / (j + 1) * (1 - sum(u[: j + 1]))) for j in range(nx)]
    rho = np.argmax(xx)
    lam = 1 / (rho + 1) * (1 - sum(u[j] for j in range(rho + 1)))
    y = np.maximum(np.zeros((nx, 1)), x + lam * np.ones((nx, 1)))

    return y


def polytopeProjection(verts, q, W=None, alpha0=None, tol=1e-10, accelerate=True):
    """
    Projects q onto the polytope defined by the convex-hull of the columns in the 2-d array verts.
    """
    nd, nv = np.shape(verts)

    if W is None:
        W = np.eye(nd)
    else:
        if not (W.shape[0] == nd and W.shape[1] == nd):
            raise "W should be square matrix of dimension consistent with the vertices of the polytope"

    it = 0
    err = 1.0

    VTWV = np.matmul(verts.T, np.matmul(W, verts))
    VTWV = 0.5 * (VTWV.T + VTWV)
    VTWq = np.matmul(verts.T, np.matmul(W, q))
    ev = np.linalg.eigvalsh(VTWV)
    L = np.max(ev)

    #  fun = lambda alpha : 0.5*np.norm(x - np.matmul(verts,alpha))**2
    def grad(alpha):
        return np.matmul(VTWV, alpha) - VTWq

    if alpha0 is None:
        alpha = 1.0 / nv * np.ones((nv, 1))
    else:
        alpha = np.reshape(alpha0, (nv, 1))
    g = grad(alpha)

    if accelerate == False:
        while not (err <= tol):
            alpha1 = simplexProjection(alpha - 1.0 / L * g)
            g = grad(alpha1)
            err = np.linalg.norm(alpha1 - alpha)
            alpha = alpha1
            it = it + 1
    else:
        alpha_1 = alpha
        while not (err <= tol):
            alphay = alpha + it / (it + 3) * (alpha - alpha_1)
            g = grad(alphay)
            alpha1 = simplexProjection(alphay - 1.0 / L * g)
            err = np.linalg.norm(alpha1 - alpha)
            alpha_1 = alpha
            alpha = alpha1
            it = it + 1

    Valpha_q = np.matmul(verts, alpha) - q
    dist = np.sqrt(np.matmul(Valpha_q.T, np.matmul(W, Valpha_q)))

    return alpha, dist[0, 0], it, err
