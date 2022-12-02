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

# # Get collocation points
# tau_root = ca.collocation_points(d, 'radau') # legendre
# [_,_,B] = ca.collocation_coeff(tau_root)#
# Bt = np.array(B)
# Ct = ca.collocation_interpolators(tau_root)#
# C = np.array(Ct[0:d+1]).T
# D = np.array(Ct[-1])

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
T = 7.5

# Declare model variables
x1 = ca.SX.sym("x1")
x2 = ca.SX.sym("x2")
x = ca.vertcat(x1, x2)
u = ca.SX.sym("u")

# Model equations
m = 1
l = 1
g = 9.81
b = 0.01
I = m * g * l**2

pdot = x2
vdot = 1.0 / I * (u - m * g * l * ca.sin(x1) - b * x2)
xdot = ca.vertcat(pdot, vdot)

# Objective term
# L = x1**2 + x2**2 + u**2
L = (x1 - ca.pi) ** 2 + x2**2 + 0.01 * u[0] ** 2

# Continuous time dynamics
f = ca.Function("f", [x, u], [xdot, L], ["x", "u"], ["xdot", "L"])

# Control discretization
N = 150  # number of control intervals
h = T / N

# Start with an empty NLP
w = []
w0 = []
lbw = []
ubw = []
J = 0
l = []
l0 = []
lbl = []
ubl = []
g = []
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

# Initial conditions
Xk = ca.MX.sym("X0", 2)
w.append(Xk)
lbw.append([0, 0.0])
ubw.append([0, 0.0])
w0.append([0, 0])
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym("U_" + str(k))
    w.append(Uk)
    lbw.append([-10])
    ubw.append([10])
    w0.append([0])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), 2)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-2 * ca.pi, -10.0])
        ubw.append([2 * ca.pi, 10.0])
        w0.append([0, 0])

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
        lbg.append([0, 0])
        ubg.append([0, 0])

        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]
        # J = J + D[j] * qj * h

        # Add contribution to quadrature function
        # J = J + B[j]*qj*h

    _, Jk = f(Xk_end, Uk)
    J = J + Jk * h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym("X_" + str(k + 1), 2)
    w.append(Xk)
    lbw.append([-2 * ca.pi, -10.0])
    ubw.append([2 * ca.pi, 10.0])
    w0.append([0, 0])
    x_plot.append(Xk)
    # Add equality constraint
    g.append(Xk_end - Xk)
    lbg.append([0, 0])
    ubg.append([0, 0])

# Impose final bound conditions
g.append([ca.pi, 0] - Xk)
lbg.append([0, 0])
ubg.append([0, 0])

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

# np.save("Results/Pendulum/x_pendulum_casadi.npy", x_opt)
# np.save("Results/Pendulum/u_pendulum_casadi.npy", u_opt)

obj = 0.0
for i in range(len(x_opt[0]) - 1):
    obj = obj + (x_opt[0][i] - ca.pi) ** 2 + x_opt[1][i] ** 2 + 0.01 * u_opt[0][i] ** 2

obj = obj * 0.05

print("Objective function value", obj)

# Plot the result
tgrid = np.linspace(0, T, N + 1)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[0], "--")
plt.plot(tgrid, x_opt[1], "-")
plt.step(tgrid, np.append(np.nan, u_opt[0]), "-.")
plt.xlabel("t")
plt.legend(["x1", "x2", "u"])
plt.grid()
plt.show()
