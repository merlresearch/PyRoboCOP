# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Plot comparison results in the Acrobot systems obtained with PyRoboCOP, Casadi, Pyomo
# Before running this code run Simulations/main_acrobot.py, Simulations/Casadi/acrobot.py Simulations/pyomo/acorbot.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"family": "normal", "size": 12}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

path = "Results/Acrobot"
# Pyrobocop
x1_pyrobocop = np.load(path + "q1_acrobot_pyrobocop.npy")
x2_pyrobocop = np.load(path + "q2_acrobot_pyrobocop.npy")
u_pyrobocop = np.load(path + "u_acrobot_pyrobocop.npy")
# Casadi
x_casadi = np.load(path + "x_acrobot_casadi.npy")
u_casadi = np.load(path + "u_acrobot_casadi.npy")
# Pyomo
x1_pyomo = np.load(path + "x1_acrobot_pyomo.npy")
x2_pyomo = np.load(path + "x2_acrobot_pyomo.npy")
u_pyomo = np.load(path + "u_acrobot_pyomo.npy")

plt.figure()

plt.plot(x1_pyrobocop, "b")
plt.plot(x_casadi[0, :], "b--")
plt.plot(x1_pyomo, "b-.")

plt.plot(x2_pyrobocop, "r")
plt.plot(x_casadi[1, :], "r--")
plt.plot(x2_pyomo, "r-.")

plt.plot(u_pyrobocop, "g")
plt.plot(u_casadi[0, :], "g--")
plt.plot(u_pyomo[1:], "g-.")

plt.xlabel("steps")
plt.legend(
    [
        "q1 pyrobocop",
        "q1 casadi",
        "q1 pyomo",
        "q2 pyrobocop",
        "q2 casadi",
        "q2 pyomo",
        "u pyrobocop",
        "u casadi",
        "u pyomo",
    ],
    loc="upper center",
)
plt.grid()
plt.savefig("traj_acrobot.pdf")

plt.figure()
plt.plot(u_pyrobocop, "b")
plt.plot(u_casadi[0, :], "r--")
plt.plot(u_pyomo[1:], "g-.")

plt.xlabel("steps")
plt.legend(["u pyrobocop", "u casadi", "u pyomo"])
plt.grid()
plt.savefig("control_acrobot.pdf")

plt.show()
