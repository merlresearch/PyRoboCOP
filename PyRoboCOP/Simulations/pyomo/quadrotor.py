# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
from Envs.Dynamics.param_dict_quadrotor import param_dict as params
from pyomo.core.expr import current as EXPR
from pyomo.dae import *
from pyomo.environ import *

# parameters
# Model equations
# Model equations

m = params["m"]
Ix = params["Ix"]
Iy = params["Iy"]
Iz = params["Iz"]
g = params["g"]

dt = 0.033
u_max = 10
v_max = 40
# time and length scales
tf = 6.6

# create a model object
m = ConcreteModel()

# define the independent variable
m.t = ContinuousSet(bounds=(0, tf))

# define control inputs
m.u1 = Var(m.t, bounds=(-u_max, u_max))
m.u2 = Var(m.t, bounds=(-u_max, u_max))
m.u3 = Var(m.t, bounds=(-u_max, u_max))
m.u4 = Var(m.t, bounds=(-u_max, u_max))

# define the dependent variables
m.xx = Var(m.t)
m.y = Var(m.t)
m.z = Var(m.t)
m.psi = Var(m.t)
m.theta = Var(m.t)
m.phi = Var(m.t)
m.xdot = Var(m.t)
m.ydot = Var(m.t)
m.zdot = Var(m.t)
m.p = Var(m.t)
m.q = Var(m.t)
m.r = Var(m.t)

# define derivatives
m.xx_dot = DerivativeVar(m.xx)
m.y_dot = DerivativeVar(m.y)
m.z_dot = DerivativeVar(m.z)
m.psi_dot = DerivativeVar(m.psi)
m.theta_dot = DerivativeVar(m.theta)
m.phi_dot = DerivativeVar(m.phi)
m.xdot_dot = DerivativeVar(m.xdot)
m.ydot_dot = DerivativeVar(m.ydot)
m.zdot_dot = DerivativeVar(m.zdot)
m.p_dot = DerivativeVar(m.p)
m.q_dot = DerivativeVar(m.q)
m.r_dot = DerivativeVar(m.r)


# define the differential equation as constrainta

# define derivatives
m.ode_xx = Constraint(m.t, rule=lambda m, t: m.xx_dot[t] == m.xdot[t])
m.ode_y = Constraint(m.t, rule=lambda m, t: m.y_dot[t] == m.ydot[t])
m.ode_z = Constraint(m.t, rule=lambda m, t: m.z_dot[t] == m.zdot[t])
m.ode_psi = Constraint(
    m.t,
    rule=lambda m, t: m.psi_dot[t]
    == m.q[t] * sin(m.phi[t]) / cos(m.theta[t]) + m.r[t] * cos(m.phi[t]) / cos(m.theta[t]),
)
m.ode_theta = Constraint(m.t, rule=lambda m, t: m.theta_dot[t] == m.q[t] * cos(m.phi[t]) - m.r[t] * sin(m.phi[t]))
m.ode_phi = Constraint(
    m.t,
    rule=lambda m, t: m.phi_dot[t]
    == m.p[t] + m.q[t] * (sin(m.phi[t]) * tan(m.theta[t])) + m.r[t] * (cos(m.phi[t]) * tan(m.theta[t])),
)
m.ode_xdot = Constraint(
    m.t,
    rule=lambda m, t: m.xdot_dot[t]
    == -1.0
    / params["m"]
    * (sin(m.phi[t]) * sin(m.psi[t]) + cos(m.phi[t]) * cos(m.psi[t]) * sin(m.theta[t]))
    * m.u1[t],
)
m.ode_ydot = Constraint(
    m.t,
    rule=lambda m, t: m.ydot_dot[t]
    == -1.0
    / params["m"]
    * (cos(m.psi[t]) * sin(m.phi[t]) - cos(m.phi[t]) * sin(m.psi[t]) * sin(m.theta[t]))
    * m.u1[t],
)
m.ode_zdot = Constraint(
    m.t,
    rule=lambda m, t: m.zdot_dot[t] == params["g"] - 1.0 / params["m"] * (cos(m.phi[t]) * cos(m.theta[t])) * m.u1[t],
)
m.ode_p = Constraint(m.t, rule=lambda m, t: m.p_dot[t] == (Iy - Iz) / Ix * m.q[t] * m.r[t] + 1.0 / Ix * m.u2[t])
m.ode_q = Constraint(m.t, rule=lambda m, t: m.q_dot[t] == (Iz - Ix) / Iy * m.p[t] * m.r[t] + 1.0 / Iy * m.u3[t])
m.ode_r = Constraint(m.t, rule=lambda m, t: m.r_dot[t] == (Ix - Iy) / Iz * m.p[t] * m.q[t] + 1.0 / Iz * m.u4[t])


# path constraints
m.path_xxl = Constraint(m.t, rule=lambda m, t: m.xx[t] >= -2 * np.pi)
m.path_xxu = Constraint(m.t, rule=lambda m, t: m.xx[t] <= 2 * np.pi)
m.path_yl = Constraint(m.t, rule=lambda m, t: m.y[t] >= -2 * np.pi)
m.path_yu = Constraint(m.t, rule=lambda m, t: m.y[t] <= 2 * np.pi)
m.path_zl = Constraint(m.t, rule=lambda m, t: m.z[t] >= -v_max)
m.path_zu = Constraint(m.t, rule=lambda m, t: m.z[t] <= v_max)
m.path_psil = Constraint(m.t, rule=lambda m, t: m.psi[t] >= -v_max)
m.path_psiu = Constraint(m.t, rule=lambda m, t: m.psi[t] <= v_max)
m.path_thetal = Constraint(m.t, rule=lambda m, t: m.theta[t] >= -v_max)
m.path_thetau = Constraint(m.t, rule=lambda m, t: m.theta[t] <= v_max)
m.path_phil = Constraint(m.t, rule=lambda m, t: m.phi[t] >= -v_max)
m.path_phiu = Constraint(m.t, rule=lambda m, t: m.phi[t] <= v_max)
m.path_xdotl = Constraint(m.t, rule=lambda m, t: m.xdot[t] >= -v_max)
m.path_xdotu = Constraint(m.t, rule=lambda m, t: m.xdot[t] <= v_max)
m.path_ydotl = Constraint(m.t, rule=lambda m, t: m.ydot[t] >= -v_max)
m.path_ydotu = Constraint(m.t, rule=lambda m, t: m.ydot[t] <= v_max)
m.path_zdotl = Constraint(m.t, rule=lambda m, t: m.zdot[t] >= -v_max)
m.path_zdotu = Constraint(m.t, rule=lambda m, t: m.zdot[t] <= v_max)
m.path_pl = Constraint(m.t, rule=lambda m, t: m.p[t] >= -v_max)
m.path_pu = Constraint(m.t, rule=lambda m, t: m.p[t] <= v_max)
m.path_ql = Constraint(m.t, rule=lambda m, t: m.q[t] >= -v_max)
m.path_qu = Constraint(m.t, rule=lambda m, t: m.q[t] <= v_max)
m.path_rl = Constraint(m.t, rule=lambda m, t: m.r[t] >= -v_max)
m.path_ru = Constraint(m.t, rule=lambda m, t: m.r[t] <= v_max)

# initial conditions
m.pc = ConstraintList()
m.pc.add(m.xx[0] == 0)
m.pc.add(m.y[0] == 0)
m.pc.add(m.z[0] == 0)
m.pc.add(m.psi[0] == 0)
m.pc.add(m.theta[0] == 0)
m.pc.add(m.phi[0] == 0)
m.pc.add(m.xdot[0] == 0)
m.pc.add(m.ydot[0] == 0)
m.pc.add(m.zdot[0] == 0)
m.pc.add(m.p[0] == 0)
m.pc.add(m.q[0] == 0)
m.pc.add(m.r[0] == 0)

# final conditions
m.pc.add(m.xx[tf] == 2.0)
m.pc.add(m.y[tf] == 2.0)
m.pc.add(m.z[tf] == 3.0)
m.pc.add(m.psi[tf] == 0)
m.pc.add(m.theta[tf] == 0)
m.pc.add(m.phi[tf] == 0)
m.pc.add(m.xdot[tf] == 0)
m.pc.add(m.ydot[tf] == 0)
m.pc.add(m.zdot[tf] == 0)
m.pc.add(m.p[tf] == 0)
m.pc.add(m.q[tf] == 0)
m.pc.add(m.r[tf] == 0)

# final conditions on the control inputs
# m.pc.add(m.av[tf]==0)
# m.pc.add(m.phi[tf]==0)

# define the optimization objective
m.integral = Integral(
    m.t,
    wrt=m.t,
    rule=lambda m, t: (m.xx[t] - 2.0) ** 2
    + (m.y[t] - 2.0) ** 2
    + (m.z[t] - 3.0) ** 2
    + m.u1[t] ** 2
    + m.u2[t] ** 2
    + m.u3[t] ** 2
    + m.u4[t] ** 2,
)
m.obj = Objective(expr=m.integral)

# transform and solve
discretizer = TransformationFactory("dae.collocation")
discretizer.apply_to(m, wrt=m.t, nfe=200, ncp=1, scheme="LAGRANGE-RADAU")

solver = SolverFactory("ipopt").solve(m, tee=True).write()


# access the results
t = np.array([t for t in m.t])

xx = np.array([m.xx[t]() for t in m.t])
y = np.array([m.y[t]() for t in m.t])
z = np.array([m.z[t]() for t in m.t])
phi = np.array([m.phi[t]() for t in m.t])
theta = np.array([m.theta[t]() for t in m.t])
psi = np.array([m.psi[t]() for t in m.t])
xdot = np.array([m.xdot[t]() for t in m.t])
ydot = np.array([m.ydot[t]() for t in m.t])
zdot = np.array([m.zdot[t]() for t in m.t])
p = np.array([m.p[t]() for t in m.t])
q = np.array([m.q[t]() for t in m.t])
r = np.array([m.r[t]() for t in m.t])
u1 = np.array([m.u1[t]() for t in m.t])
u2 = np.array([m.u2[t]() for t in m.t])
u3 = np.array([m.u3[t]() for t in m.t])
u4 = np.array([m.u4[t]() for t in m.t])


np.save("Results/Quadrotor/xx_quadrotor_pyomo.npy", xx)
np.save("Results/Quadrotor/y_quadrotor_pyomo.npy", y)
np.save("Results/Quadrotor/z_quadrotor_pyomo.npy", z)
np.save("Results/Quadrotor/u1_quadrotor_pyomo.npy", u1)
np.save("Results/Quadrotor/u2_quadrotor_pyomo.npy", u2)
np.save("Results/Quadrotor/u3_quadrotor_pyomo.npy", u3)
np.save("Results/Quadrotor/u4_quadrotor_pyomo.npy", u4)


def plot_results(t, xx, y, z, u1, u2, u3, u4):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(t, xx, t, y, t, z, t, u1, t, u2, t, u3, t, u4)
    ax.legend(["xx", "y", "z", "u1", "u2", "u3", "u4"])
    ax.grid(True)


plot_results(t, xx, y, z, u1, u2, u3, u4)

plt.show()
