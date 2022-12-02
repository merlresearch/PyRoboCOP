# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import os.path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyomo.core.expr import current as EXPR
from pyomo.dae import *
from pyomo.environ import *

# parameters
# Model equations
m1 = 0.5
m2 = 0.5
L1 = 0.5
L2 = 0.5
lc1 = 0.2
lc2 = 0.2
I1 = 0.5
I2 = 0.5
g = 9.81
dt = 0.05
u_max = 20
v_max = 20
# time and length scales
tf = 5.0

# create a model object
m = ConcreteModel()

# define the independent variable
m.t = ContinuousSet(bounds=(0, tf))

# define control inputs
m.u = Var(m.t, bounds=(-u_max, u_max))

# define the dependent variables
m.q1 = Var(m.t)
m.q2 = Var(m.t)
m.v1 = Var(m.t)
m.v2 = Var(m.t)

# define derivatives
m.q1_dot = DerivativeVar(m.q1)
m.q2_dot = DerivativeVar(m.q2)
m.v1_dot = DerivativeVar(m.v1)
m.v2_dot = DerivativeVar(m.v2)


# define the differential equation as constrainta
def fd(m, t):
    M11 = I1 + I2 + m2 * L1**2 + 2 * m2 * L1 * lc2 * cos(m.q2[t])
    M12 = I2 + m2 * L1 * lc2 * cos(m.q2[t])
    M21 = I2 + m2 * L1 * lc2 * cos(m.q2[t])
    M22 = I2

    C11 = -2 * m2 * L1 * lc2 * sin(m.q2[t]) * m.v2[t]
    C12 = -m2 * L1 * lc2 * sin(m.q2[t]) * m.v2[t]
    C21 = m2 * L1 * lc2 * sin(m.q2[t]) * m.v1[t]
    C22 = 0

    g1 = (m1 * lc1 + m2 * L1) * g * sin(m.q1[t]) + m2 * g * L2 * sin(m.q1[t] + m.q2[t])
    g2 = m2 * g * L2 * sin(m.q1[t] + m.q2[t])

    # Coriolis, gravitational and input component
    n1 = -(C11 * m.v1[t] + C12 * m.v2[t] + g1)
    n2 = -(C21 * m.v1[t] + C22 * m.v2[t] + g2) + m.u[t]
    # qddpt = M^-1(-C*qdot - G + U)
    return [(M22 * n1 - M12 * n2) / (M11 * M22 - M21 * M12), (-M21 * n1 + M11 * n2) / (M11 * M22 - M21 * M12)]


m.ode_q1 = Constraint(m.t, rule=lambda m, t: m.q1_dot[t] == m.v1[t])
m.ode_q2 = Constraint(m.t, rule=lambda m, t: m.q2_dot[t] == m.v2[t])
m.ode_v1 = Constraint(m.t, rule=lambda m, t: m.v1_dot[t] == fd(m, t)[0])
m.ode_v2 = Constraint(m.t, rule=lambda m, t: m.v2_dot[t] == fd(m, t)[1])

# path constraints
m.path_q1l = Constraint(m.t, rule=lambda m, t: m.q1[t] >= -2 * np.pi)
m.path_q1u = Constraint(m.t, rule=lambda m, t: m.q1[t] <= 2 * np.pi)
m.path_q2l = Constraint(m.t, rule=lambda m, t: m.q2[t] >= -2 * np.pi)
m.path_q2u = Constraint(m.t, rule=lambda m, t: m.q2[t] <= 2 * np.pi)
m.path_v1l = Constraint(m.t, rule=lambda m, t: m.v1[t] >= -v_max)
m.path_v1u = Constraint(m.t, rule=lambda m, t: m.v1[t] <= v_max)
m.path_v2l = Constraint(m.t, rule=lambda m, t: m.v2[t] >= -v_max)
m.path_v2u = Constraint(m.t, rule=lambda m, t: m.v2[t] <= v_max)

# initial conditions
m.pc = ConstraintList()
m.pc.add(m.q1[0] == 0)
m.pc.add(m.q2[0] == 0)
m.pc.add(m.v1[0] == 0)
m.pc.add(m.v2[0] == 0)
m.pc.add(m.u[0] == 0)

# final conditions
m.pc.add(m.q1[tf] == np.pi)
m.pc.add(m.q2[tf] == 0)
m.pc.add(m.v1[tf] == 0)
m.pc.add(m.v2[tf] == 0)

# final conditions on the control inputs
# m.pc.add(m.av[tf]==0)
# m.pc.add(m.phi[tf]==0)

# define the optimization objective
m.integral = Integral(
    m.t,
    wrt=m.t,
    rule=lambda m, t: (m.q1[t] - np.pi) ** 2
    + m.q2[t] ** 2
    + 0.1 * m.v1[t] ** 2
    + 0.1 * m.v2[t] ** 2
    + 0.1 * m.u[t] ** 2,
)
m.obj = Objective(expr=m.integral)

# transform and solve
discretizer = TransformationFactory("dae.collocation")
discretizer.apply_to(m, wrt=m.t, nfe=100, ncp=1, scheme="LAGRANGE-RADAU")

solver = SolverFactory("ipopt").solve(m, tee=True).write()


# access the results
t = np.array([t for t in m.t])

q1 = np.array([m.q1[t]() for t in m.t])
q2 = np.array([m.q2[t]() for t in m.t])
v1 = np.array([m.v1[t]() for t in m.t])
v2 = np.array([m.v2[t]() for t in m.t])
u = np.array([m.u[t]() for t in m.t])


np.save("Results/Acrobot/x1_acrobot_pyomo.npy", q1)
np.save("Results/Acrobot/x2_acrobot_pyomo.npy", q2)
np.save("Results/Acrobot/u_acrobot_pyomo.npy", u)


def plot_results(t, q1, q2, u):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(t, q1, t, q2, t, u)
    ax.legend(["q1", "q2", "Torque"])
    ax.grid(True)

    # ax[1].plot(t, phi, t, a)
    # ax[1].legend(['Wheel Position', 'Car Direction'])
    #
    # ax[2].plot(t, v)
    # ax[2].legend(['Velocity'])
    # ax[2].set_ylabel('m/s')
    # for axes in ax:
    #     axes.grid(True)


plot_results(t, q1, q2, u)

# scl = 0.3
#
#
# def draw_car(x=0, y=0, a=0, phi=0):
#     R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
#     car = np.array([[0.2, 0.5], [-0.2, 0.5], [0, 0.5], [0, -0.5],
#                     [0.2, -0.5], [-0.2, -0.5], [0, -0.5], [0, 0], [L, 0], [L, 0.5],
#                     [L + 0.2 * np.cos(phi), 0.5 + 0.2 * np.sin(phi)],
#                     [L - 0.2 * np.cos(phi), 0.5 - 0.2 * np.sin(phi)], [L, 0.5], [L, -0.5],
#                     [L + 0.2 * np.cos(phi), -0.5 + 0.2 * np.sin(phi)],
#                     [L - 0.2 * np.cos(phi), -0.5 - 0.2 * np.sin(phi)]])
#     carz = scl * R.dot(car.T)
#     plt.plot(x + carz[0], y + carz[1], 'k', lw=2)
#     plt.plot(x, y, 'k.', ms=10)
#
#
# eradicate: no # plt.figure(figsize=(10, 10))
# eradicate: no # for xs, ys, ts, ps in zip(x, y, a, phi):
# eradicate: no #     draw_car(xs, ys, ts, scl * ps)
# eradicate: no # plt.plot(x, y, 'r--', lw=0.8)
# eradicate: no # plt.axis('square')
# eradicate: no # plt.grid(True)
plt.show()
