# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
from pyomo.dae import *
from pyomo.environ import *

# parameters
u_max = 10
m_p = 1  # Mass
l = 1  # length
b = 0.01  # Friction coefficient
g = 9.81  # gravity
I = m_p * g * l**2

# time and length scales
tf = 7.5

# create a model object
m = ConcreteModel()

# define the independent variable
m.t = ContinuousSet(bounds=(0, tf))

# define control inputs
m.u = Var(m.t, bounds=(-u_max, u_max))

# define the dependent variables
m.x = Var(m.t)
m.v = Var(m.t)

# define derivatives
m.x_dot = DerivativeVar(m.x)
m.v_dot = DerivativeVar(m.v)


# define the differential equation as constrainta
m.ode_x = Constraint(m.t, rule=lambda m, t: m.x_dot[t] == m.v[t])
m.ode_v = Constraint(m.t, rule=lambda m, t: m.v_dot[t] == 1.0 / I * (m.u[t] - m_p * g * l * sin(m.x[t]) - b * m.v[t]))

# path constraints
m.path_x1 = Constraint(m.t, rule=lambda m, t: m.x[t] >= -2 * np.pi)
m.path_x2 = Constraint(m.t, rule=lambda m, t: m.x[t] <= 2 * np.pi)
m.path_v1 = Constraint(m.t, rule=lambda m, t: m.v[t] >= -10)
m.path_v2 = Constraint(m.t, rule=lambda m, t: m.v[t] <= 10)

# initial conditions
m.pc = ConstraintList()
m.pc.add(m.x[0] == 0)
m.pc.add(m.v[0] == 0)
m.pc.add(m.u[0] == 0)  # maybe

# final conditions
m.pc.add(m.x[tf] == np.pi)
m.pc.add(m.v[tf] == 0)

# final conditions on the control inputs
# m.pc.add(m.av[tf]==0)
# m.pc.add(m.phi[tf]==0)

# define the optimization objective
m.integral = Integral(m.t, wrt=m.t, rule=lambda m, t: (m.x[t] - np.pi) ** 2 + m.v[t] ** 2 + 0.01 * m.u[t] ** 2)
m.obj = Objective(expr=m.integral)

# transform and solve
discretizer = TransformationFactory("dae.collocation")
discretizer.apply_to(m, wrt=m.t, nfe=150, ncp=2, scheme="LAGRANGE-LEGENDRE")  # RADAU LEGENDRE

solver = SolverFactory("ipopt")
solver.solve(m, tee=True).write()


# access the results
t = np.array([t for t in m.t])

x = np.array([m.x[t]() for t in m.t])
v = np.array([m.v[t]() for t in m.t])
u = np.array([m.u[t]() for t in m.t])

np.save("Results/Pendulum/x_pendulum_pyomo.npy", x)
np.save("Results/Pendulum/v_pendulum_pyomo.npy", v)
np.save("Results/Pendulum/u_pendulum_pyomo.npy", u)


def plot_results(t, x, v, u):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(t, x, t, v, t, u)
    ax.legend(["Theta", "Velocity", "Torque"])
    ax.grid(True)


plot_results(t, x, v, u)

plt.show()
