# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import collocation as co
import numpy as np
import scipy as sc


def solve_integral(fun, xmin, xmax, nfe, colloc):
    funint = 0.0
    dx = (xmax - xmin) / nfe
    for i in range(nfe):
        for j in range(colloc.ncolloc):
            xij = xmin + i * dx + colloc.rx[j + 1] * dx
            funint = funint + dx * colloc.lagpolyobjint1[j] * fun(xij)
    return funint


def compare_solve_integral_schemes(fun, xmin, xmax, nfe):
    roots = ["legendre", "radau"]
    for r in roots:
        for nc in range(1, 6):
            colloc = co.collocation(nc, r)
            funint = solve_integral(fun, xmin, xmax, nfe, colloc)
            print("Integral: ", r, " nc: ", nc, " ", funint)


nfe = 10  # number of finite elements
# we will intergrate sin(x) from 0 to pi/2
def fun(x):
    return np.sin(x)


print("*** Integrating sin(x) from 0 to pi/2 ***")
compare_solve_integral_schemes(fun, 0, np.pi / 2, nfe)

# we will intergrate exp(x) from 0 to 2


def fun(x):
    return np.exp(x)


print("*** Integrating exp(x) from 0 to 2 ***")
compare_solve_integral_schemes(fun, 0, 2.0, nfe)

# we will intergrate 1/x^2 from 1 to 10


def fun(x):
    return 1 / x**2


print("*** Integrating exp(x) from 1 to 10 ***")
compare_solve_integral_schemes(fun, 1.0, 10.0, nfe)
