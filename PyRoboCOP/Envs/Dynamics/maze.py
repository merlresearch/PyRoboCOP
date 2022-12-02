#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.CircularMaze.utils_gp as utils_gp
import matplotlib.pyplot as plt
import numpy as np
from Envs.Dynamics.CircularMaze.Maze_GP import GaussianProcessMazeModel as MazeGP


class Maze(object):
    def __init__(
        self,
        ring=1,
        coeff_dict=None,
        cost_dict=None,
        file_name="Exploration_data_maze_sum_sines_freq1bang5_run2theta_acc_acaus",
        use_PP=None,
        pp_frac=0.3,
    ):

        # Dynamical parameters of the table platform that moves the maze
        if coeff_dict == None:
            model_beta_dot = np.array(
                [-1.50222372, 0.931466148, -1.09510221e-01, -9.24299235e-04, 1.54154966e00, -4.32335212e-02]
            )
            model_gamma_dot = np.array([-0.08791515, -0.00451278, -1.37314616, 0.9334687, 0.03390455, 1.45394861])

            coeff_dict = {
                "coeff_betaddot": 1,
                "coeff_gammaddot": 1,
                "table_friction_factor": 0,
                "dt": 1.0 / 30,
                "beta_model_coefficients": model_beta_dot,
                "gamma_model_coefficients": model_gamma_dot,
            }
        # This is added to the class to get the ring-dependent coefficients
        self.coeff_dict = coeff_dict

        # Cost parameters that are required to define the objective function
        if cost_dict == None:
            dX = 6
            dU = 2
            # state_cost_coeff: the coefficient with which we weight the cost for each state component
            state_cost_coeff = np.zeros((dX,))
            state_cost_coeff[0] = 1
            state_cost_coeff[1] = 0.1
            # control_cost_coeff: coefficient with which we weight the cost for each control component
            control_cost_coeff = np.zeros((dU,))
            control_cost_coeff[0] = 1
            control_cost_coeff[1] = 1

            if ring == 1:
                self.init_theta = 0
                target_theta = np.pi / 4
                self.max_amp_init_sin = 1
                self.max_amp_init_cos0 = -0.05
                self.max_amp_init_cos1 = 0.01
            elif ring == 2:
                self.init_theta = np.pi / 4
                target_theta = 0
                self.max_amp_init_sin = 1
                self.max_amp_init_cos = 0.01
            elif ring == 3:
                self.init_theta = 0
                target_theta = np.pi / 4
                self.max_amp_init_sin = 1
                self.max_amp_init_cos = -0.05
            elif ring == 4:
                self.init_theta = np.pi / 4
                target_theta = 0
                self.max_amp_init_sin = 1
                self.max_amp_init_cos = 0.01

            print("target theta: ", target_theta)

            cost_dict = {
                "u_alpha": 0.2,
                "max_control_amp": 0.1,
                "state_cost_coeff": state_cost_coeff,
                "input_cost_coeff": control_cost_coeff,
                "target_state": target_theta,
            }
        # This is added to the class to get the ring dependent coefficients
        self.cost_dict = cost_dict
        # Load the model of the maze
        self.flg_model = "SP"  # SP for semiparametric model
        if ring == 4:
            file_name = "Exploration_data_maze_sum_sines_freq1bang5theta_acc_acausall"
        self.file_name = file_name  # Load data for the model
        self.ring = ring
        self.use_PP = use_PP
        self.pp_frac = pp_frac

        location_model = "Envs/Dynamics/CircularMaze/Models/"
        param_dict = utils_gp.get_GP_estimator_SP_fromdir(location_model, self.flg_model, self.ring, self.file_name)

        # Object that represent the maze model learned with GPR
        self.MazeGP = MazeGP(param_dict, self.coeff_dict, self.cost_dict, use_PP=self.use_PP, pp_frac=self.pp_frac)

        dt = 0.033
        self.param_dict = param_dict
        self.param_dict["dt"] = dt
        self.param_dict["g"] = 9.81

        # Parameters required by the OCP
        self.n_d = self.MazeGP.dX
        self.n_a = 0
        self.n_u = self.MazeGP.dU
        self.n_p = 0
        self.n_cc = 0  # NEW
        self.T = 90
        self.times = self.param_dict["dt"] * np.array(list(range(self.T + 1)))  # NEW
        self.nnz_jac = 64
        self.nnz_hess = 0

    def get_model_params(self):
        return self.param_dict["dt"]

    def get_info(self):
        """
        Method to return OCP info
        n_d - number of differential vars
        n_a - number of algebraic vars
        n_u - number or controls vars
        n_p - number of parameters
        n_cc - number of complementarity variables (part of algebraic vars)
        T   - number of time-steps
        times - the time at start of each of the time intervals, an array of (T+1)
        nnz_jac - number of nonzeros in jacobian of DAE
        nnz_hess - number of nonzeros in hessian of OCP at each time-step
        """
        return self.n_d, self.n_a, self.n_u, self.n_p, self.n_cc, self.T, self.times, self.nnz_jac, self.nnz_hess

    def bounds(self, t):
        """
        Method to return the bounds information on the OCP instance
        lb = [xlb,xdotlb,ylb,ulb]
        ub = [xub,xdotub,yub,uub]
        """
        u_max = self.cost_dict["max_control_amp"]
        lb = np.array(
            [
                -2 * np.pi,
                -10.0,
                -2 * np.pi,
                -10.0,
                -2 * np.pi,
                -10.0,
                -1.0e30,
                -1.0e30,
                -1.0e30,
                -1.0e30,
                -1.0e30,
                -1.0e30,
                -u_max,
                -u_max,
            ]
        )
        ub = np.array(
            [
                2 * np.pi,
                10.0,
                2 * np.pi,
                10.0,
                2 * np.pi,
                10.0,
                1.0e30,
                1.0e30,
                1.0e30,
                1.0e30,
                1.0e30,
                1.0e30,
                u_max,
                u_max,
            ]
        )
        return lb, ub

    def initialpoint(self, t):
        """
        Method to return the initial guess for the OCP instance
        """
        x0 = np.zeros((self.n_d,))
        xdot0 = np.zeros((self.n_d,))
        u0 = np.zeros((self.n_u,))
        # To ADD an initialization you can uncomment these lines
        # x0[0] = self.sin_input_profile(max_amp=self.max_amp_init_sin,T=self.T,dt=t)
        # u0[0]=self.cos_input_profile(max_amp=self.max_amp_init_cos0,T=self.T,dt=t)
        # u0[1]=self.cos_input_profile(max_amp=self.max_amp_init_cos1,T=self.T,dt=t)
        y0 = np.array([])
        return x0, xdot0, y0, u0

    def initialcondition(self):
        """
        Method to return the initial condition for the OCP instance
        """
        xic = np.zeros((self.n_d,))
        xic[0] = self.init_theta
        # Impose the initial condition also in the Maze Env
        self.MazeGP.reset(np.array(xic))
        return xic

    def objective(self, t, x, xdot, y, u, params):
        """
        Method to return the objective function of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        c = self.MazeGP.cost(x, u)
        return c

    # def gradient(self, t, x, xdot, y, u, params):
    #     """
    #     Method to return the gradient of the objective function of the OCP instance
    #     x - numpy 1-d array of differential variables at time t
    #     xdot - numpy 1-d array of time derivative differential variables at time t
    #     y - numpy 1-d array of algebraic variables at time t
    #     u - numpy 1-d array of control avriables at time t
    #     params - parameters
    #     """
    #     x_u = np.concatenate((x, u), axis=0).reshape(-1, 1)
    #     # J = self.cost_jacobian(x_u).squeeze()
    #     J = self.MazeGP.gradient_cost(x_u)
    #
    #     return J
    #
    def constraint(self, c, t, x, xdot, y, u, params):
        """
        Method to return the constraint of the OCP instance
        x - numpy 1-d array of differential variables at time t
        xdot - numpy 1-d array of time derivative differential variables at time t
        y - numpy 1-d array of algebraic variables at time t
        u - numpy 1-d array of control avriables at time t
        params - parameters
        """
        discrete_step = self.MazeGP.forward_dynamics(x, u)
        c[:, 0] = -xdot + (discrete_step - x) / self.param_dict["dt"]

    def cos_input_profile(self, max_amp, T, dt):
        f = np.pi / 2
        omega = 2 * np.pi * f
        u = max_amp * (np.cos(f * T * dt))
        return u

    def sin_input_profile(self, max_amp, T, dt):
        f = np.pi / 2 / T
        omega = 2 * np.pi * f
        u = max_amp * (np.sin(f * T * dt))
        return u

    def step_input_profile(self, max_amp, T, dt):
        f = np.pi / 2 / T
        f = 2 * np.pi * f
        u = max_amp * np.sign(np.sin(f * T * dt))
        return u


if __name__ == "__main__":
    T = 90
    dt = 0.033
    times = dt * np.array(list(range(T + 1)))

    def sin_input_profile(max_amp, T, dt):
        # u = np.zeros((dU,))
        f = np.pi / 2 / T
        omega = 2 * np.pi * f
        u = max_amp * (np.cos(f * T * dt))
        # u[1] = -max_amp * (np.cos(f * T * dt))
        return u

    sin_profile = np.zeros((T + 1,))
    for i, t in enumerate(times):
        sin_profile[i] = sin_input_profile(max_amp=1, T=T, dt=t)

    plt.figure()
    plt.plot(times, sin_profile)
    plt.show()
