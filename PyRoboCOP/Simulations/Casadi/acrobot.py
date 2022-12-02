# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

#
#
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

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

print("D: ", D)
print("B: ", B)

# Time horizon
T = 5.0

# Declare model variables
x1 = ca.SX.sym("x1")
x2 = ca.SX.sym("x2")
x3 = ca.SX.sym("x3")
x4 = ca.SX.sym("x4")
x = ca.vertcat(x1, x2, x3, x4)
u = ca.SX.sym("u")
state_dim = 4
u_max = 10
v_max = 40
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

M11 = I1 + I2 + m2 * L1**2 + 2 * m2 * L1 * lc2 * ca.cos(x2)
M12 = I2 + m2 * L1 * lc2 * ca.cos(x2)
M21 = I2 + m2 * L1 * lc2 * ca.cos(x2)
M22 = I2

C11 = -2 * m2 * L1 * lc2 * ca.sin(x2) * x4
C12 = -m2 * L1 * lc2 * ca.sin(x2) * x4
C21 = m2 * L1 * lc2 * ca.sin(x2) * x3
C22 = 0

g1 = (m1 * lc1 + m2 * L1) * g * ca.sin(x1) + m2 * g * L2 * ca.sin(x1 + x2)
g2 = m2 * g * L2 * ca.sin(x1 + x2)

M = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M21, M22))
M_inv = ca.inv(M)
CC = ca.vertcat(ca.horzcat(C11, C12), ca.horzcat(C21, C22))
G = ca.vertcat(g1, g2)
BB = ca.vertcat(0, 1)

qdot = ca.vertcat(x3, x4)

# Generalized manipulator dynamics.
# fd = ca.solve(M, -ca.mtimes(CC, qdot) - G + BB*u)
fd = ca.mtimes(M_inv, -ca.mtimes(CC, qdot) - G + BB * u)

xdot = ca.vertcat(x3, x4, fd)

# Objective term
# L = (x1-ca.pi)*(x1-ca.pi) + x2*x2 + 0.1*x3*x3 + 0.1*x4*x4 + 0.1*u*u
L = (x1 - ca.pi) ** 2 + x2**2 + 0.1 * x3**2 + 0.1 * x4**2 + 0.1 * u**2

# Continuous time dynamics
f = ca.Function("f", [x, u], [xdot, L], ["x", "u"], ["xdot", "L"])

# Control discretization
N = 100  # number of control intervals
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
lbw.append([0, 0.0, 0, 0.0])
ubw.append([0, 0.0, 0, 0.0])
w0.append([0, 0, 0, 0.0])
x_plot.append(Xk)

# Formulate the NLP
# u_pyrobocop = np.load("../../../../Results/Acrobot/u_acrobot_pyrobocop.npy")

for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym("U_" + str(k))
    w.append(Uk)
    lbw.append([-u_max])
    ubw.append([u_max])
    w0.append([0.0])
    # w0.append([u_pyrobocop[k]])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), state_dim)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-2 * ca.pi, -2 * ca.pi, -v_max, -v_max])
        ubw.append([2 * ca.pi, 2 * ca.pi, v_max, v_max])
        w0.append([0, 0, 0, 0])

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
        lbg.append([0, 0, 0, 0])
        ubg.append([0, 0, 0, 0])

        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]

        # Add contribution to quadrature function
        # J = J + B[j]*qj*h

    _, Jk = f(Xk_end, Uk)
    J = J + Jk * h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym("X_" + str(k + 1), state_dim)
    w.append(Xk)
    lbw.append([-2 * ca.pi, -2 * ca.pi, -v_max, -v_max])
    ubw.append([2 * ca.pi, 2 * ca.pi, v_max, v_max])
    w0.append([0, 0, 0, 0])
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end - Xk)
    lbg.append([0, 0, 0, 0])
    ubg.append([0, 0, 0, 0])

# Impose final bounds constraints
g.append([ca.pi, 0, 0, 0] - Xk)
lbg.append([0, 0, 0, 0])
ubg.append([0, 0, 0, 0])
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

np.save("Results/Acrobot/x_acrobot_casadi.npy", x_opt)
np.save("Results/Acrobot/u_acrobot_casadi.npy", u_opt)

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
