# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from casadi import cos as c
from casadi import sin as s
from casadi import tan as t
from Envs.Dynamics.param_dict_quadrotor import param_dict as params

# Collocation points and coefficients
# Degree of interpolating polynomial
d = 1
# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, "radau"))  # legendre
# Coefficients of the collocation equation
C = np.zeros((d + 1, d + 1))
# Coefficients of the continuity equation
D = np.zeros(d + 1)
# Coefficients of the quadrature function
B = np.zeros(d + 1)

# Construct polynomial basis
for j in range(d + 1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d + 1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)
    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d + 1):
        C[j, r] = pder(tau_root[r])
    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)

# Time horizon
T = 6.6

# Declare model variables
xx = ca.SX.sym("x1")
y = ca.SX.sym("x2")
z = ca.SX.sym("x3")
psi = ca.SX.sym("x4")
theta = ca.SX.sym("x5")
phi = ca.SX.sym("x6")
xdot = ca.SX.sym("x7")
ydot = ca.SX.sym("x8")
zdot = ca.SX.sym("x9")
p = ca.SX.sym("x10")
q = ca.SX.sym("x11")
r = ca.SX.sym("x12")

x = ca.vertcat(xx, y, z, psi, theta, phi, xdot, ydot, zdot, p, q, r)
u1 = ca.SX.sym("u1")
u2 = ca.SX.sym("u2")
u3 = ca.SX.sym("u3")
u4 = ca.SX.sym("u4")
u = ca.vertcat(u1, u2, u3, u4)

state_dim = 12
u_dim = 4
u_max = 10
v_max = 40
# Model equations

m = params["m"]
Ix = params["Ix"]
Iy = params["Iy"]
Iz = params["Iz"]
g = params["g"]

# Generalized manipulator dynamics.
dxxdt = xdot
dydt = ydot
dzdt = zdot
dpsidt = q * s(phi) / c(theta) + r * c(phi) / c(theta)
dthetadt = q * c(phi) - r * s(phi)
dphidt = p + q * (s(phi) * t(theta)) + r * (c(phi) * t(theta))
dxdotdt = -1.0 / params["m"] * (s(phi) * s(psi) + c(phi) * c(psi) * s(theta)) * u1
dydotdt = -1.0 / params["m"] * (c(psi) * s(phi) - c(phi) * s(psi) * s(theta)) * u1
dzdotdt = params["g"] - 1.0 / params["m"] * (c(phi) * c(theta)) * u1
dpdt = (Iy - Iz) / Ix * q * r + 1.0 / Ix * u2
dqdt = (Iz - Ix) / Iy * p * r + 1.0 / Iy * u3
drdt = (Ix - Iy) / Iz * p * q + 1.0 / Iz * u4

xdot = ca.vertcat(dxxdt, dydt, dzdt, dpsidt, dthetadt, dphidt, dxdotdt, dydotdt, dzdotdt, dpdt, dqdt, drdt)

# Objective term
# L = (x1-ca.pi)*(x1-ca.pi) + x2*x2 + 0.1*x3*x3 + 0.1*x4*x4 + 0.1*u*u
L = (xx - 2.0) ** 2 + (y - 2.0) ** 2 + +((z - 3.0) ** 2) + u1**2 + u2**2 + u3**2 + u4**2

# Continuous time dynamics
f = ca.Function("f", [x, u], [xdot, L], ["x", "u"], ["xdot", "L"])

# Control discretization
N = 200  # number of control intervals
h = T / N

# Start with an empty NLP
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

# Initial conditions
Xk = ca.MX.sym("X0", state_dim)
w.append(Xk)
lbw.append([0.0] * state_dim)
ubw.append([0.0] * state_dim)
w0.append([0.0] * state_dim)
x_plot.append(Xk)

# Formulate the NLP
# u_pyrobocop = np.load("Results/Acrobot/u_acrobot_pyrobocop.npy")

for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym("U_" + str(k), u_dim)
    w.append(Uk)
    lbw.append([-u_max] * u_dim)
    ubw.append([u_max] * u_dim)
    w0.append([0.0] * u_dim)
    # w0.append([u_pyrobocop[k]])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), state_dim)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-10.0] * state_dim)
        ubw.append([10.0] * state_dim)
        w0.append([0.0] * state_dim)

    # Loop over collocation points
    Xk_end = D[0] * Xk
    for j in range(1, d + 1):
        # Expression for the state derivative at the collocation point
        xp = C[0, j] * Xk
        for r in range(d):
            xp = xp + C[r + 1, j] * Xc[r]

        # Append collocation equations
        fj, qj = f(Xc[j - 1], Uk)
        g.append(h * fj - xp)
        lbg.append([0.0] * state_dim)
        ubg.append([0.0] * state_dim)

        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]

        # Add contribution to quadrature function
        # J = J + B[j]*qj*h

    _, Jk = f(Xk_end, Uk)
    J = J + Jk * h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym("X_" + str(k + 1), state_dim)
    w.append(Xk)
    lbw.append([-10.0] * state_dim)
    ubw.append([10.0] * state_dim)
    w0.append([0.0] * state_dim)
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end - Xk)
    lbg.append([0.0] * state_dim)
    ubg.append([0.0] * state_dim)

# Impose final bounds constraints
g.append([2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] - Xk)
lbg.append([0.0] * state_dim)
ubg.append([0.0] * state_dim)
# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# Create an NLP solver
prob = {"f": J, "x": w, "g": g}
solver = ca.nlpsol("solver", "ipopt", prob)

# Function to get x and u trajectories from w
trajectories = ca.Function("trajectories", [w], [x_plot, u_plot], ["w"], ["x", "u"])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

x_opt, u_opt = trajectories(sol["x"])
x_opt = x_opt.full()  # to numpy array
u_opt = u_opt.full()  # to numpy array

obj = 0.0
for i in range(len(x_opt[0]) - 1):
    obj = (
        obj
        + (x_opt[0][i] - ca.pi) ** 2
        + x_opt[1][i] ** 2
        + 0.1 * x_opt[2][i] ** 2
        + 0.1 * x_opt[3][i] ** 2
        + 0.1 * u_opt[0][i] ** 2
    )

obj = obj * 0.05

print("Objective function value", obj)

np.save("../../../../Results/Quadrotor/x_quadrotor_casadi.npy", x_opt)
np.save("../../../../Results/Quadrotor/u_quadrotor_casadi.npy", u_opt)

# Plot the result
tgrid = np.linspace(0, T, N + 1)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[0], "--")
plt.plot(tgrid, x_opt[1], "-")
plt.plot(tgrid, x_opt[2], "--")
plt.plot(tgrid, x_opt[3], "-")
plt.step(tgrid, np.append(np.nan, u_opt[0]), "-.")
plt.xlabel("t")
plt.legend(["x1", "x2", "x3", "x4", "u"])
plt.grid()
plt.show()
