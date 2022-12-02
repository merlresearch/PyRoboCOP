# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Class modeling a static, ellipsoidal obstacle
"""


import numpy as np


class staticObstacle(object):
    """
    Ellipsoid is defined by: (q - qc)^T*W*(q-qc) <= r
    """

    def __init__(self, qc, W, r):
        self.qc = np.reshape(qc, (len(qc), 1))
        self.W = W
        self.r = r

        self.nd = len(qc)

        if not (np.shape(W)[0] == self.nd) or not (np.shape(W)[1] == self.nd):
            raise "W matrix should be compatible with dimension of qc"

        evals = np.linalg.eigvalsh(self.W)
        if not (min(evals) >= 0.0):
            raise "W matrix defining the ellipsoid should be positive semidefinite"

    def get_radius(self):
        return self.r
