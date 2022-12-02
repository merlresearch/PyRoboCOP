# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"family": "normal", "size": 12}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Pyrobocop
x_pyrobocop = np.load("q_pendulum_pyrobocop.npy")
u_pyrobocop = np.load("u_pendulum_pyrobocop.npy")
# Casadi
x_casadi = np.load("x_pendulum_casadi.npy")
u_casadi = np.load("u_pendulum_casadi.npy")
# Pyomo
x_pyomo = np.load("x_pendulum_pyomo.npy")
u_pyomo = np.load("u_pendulum_pyomo.npy")

plt.figure()

plt.plot(x_pyrobocop, "b")
plt.plot(x_casadi[0, :], "r--")
plt.plot(x_pyomo, "g-.")
plt.xlabel("steps")
plt.legend(["q pyrobocop", "q casadi", "q pyomo"])
plt.grid()
plt.savefig("traj_pendulum.pdf")

plt.figure()
plt.plot(u_pyrobocop, "b")
plt.plot(u_casadi[0, :], "r--")
plt.plot(u_pyomo[1:], "g-.")

plt.xlabel("steps")
plt.legend(["u pyrobocop", "u casadi", "u pyomo"])
plt.grid()
plt.savefig("control_pendulum.pdf")

plt.show()
