# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np


def angle_diff(q2, q1):

    dq = (q2 - q1 + np.pi) % (2.0 * np.pi) - np.pi
    return dq


def rmat_twod(ang, opt=0):

    if opt == 0:

        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        return R

    elif opt == 1:

        dR = -np.array([[np.sin(ang), np.cos(ang)], [-np.cos(ang), np.sin(ang)]])
        return dR

    elif opt == 2:

        dR2 = -np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

        return dR2

    else:
        raise RuntimeError("incorrect option")


def transform_points_twod(pts, cnt, ang):
    """pts is (2,n) numpy array
    cent is (2,) numpy array
    angle is scalar in radians"""

    R = rmat_twod(ang)
    rotated_pts = np.dot(R, pts)

    return (rotated_pts.T + cnt).T


def com_motion_from_pure_rotation_about_xy(dx, dy, tht):
    """dx, dy is position or COR w.r.t to COM"""
    """ tht is angle of rotation """

    dx_com = -dx * np.cos(tht) + dy * np.sin(tht) + dx
    dy_com = -1.0 * (dx * np.sin(tht) + dy * np.cos(tht)) + dy

    return dx_com, dy_com, tht
