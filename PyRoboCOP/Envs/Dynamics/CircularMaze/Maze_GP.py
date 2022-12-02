# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import Envs.Dynamics.CircularMaze.utils_gp as utils_gp
import numpy as np


class GaussianProcessMazeModel(object):
    def __init__(
        self, param_dict=None, coeff_dict=None, cost_dict=None, return_system_matrix=True, use_PP=None, pp_frac=0.3
    ):

        ## get the parameters that will be used in Gaussian processes

        self.X = param_dict["X"]

        # self.X=self.X[0:1000,:]
        self.std_X_NP = param_dict["std_X_NP"]
        self.std_X_PP = param_dict["std_X_PP"]
        self.std_Y = param_dict["std_Y"]
        self.length_scales = param_dict["length_scales"]

        print("length_scales", self.length_scales)
        self.scaling_factor = param_dict["scaling_factor"]
        self.w_hat = param_dict["w_hat"]
        self.alpha_hat = param_dict["alpha_hat"]

        self.beta_model_coefficients = coeff_dict["beta_model_coefficients"]
        self.gamma_model_coefficients = coeff_dict["gamma_model_coefficients"]

        self._size_X = self.X.shape

        self.dt = coeff_dict["dt"]
        self.coeff_betaddot = coeff_dict["coeff_betaddot"]
        self.coeff_gammaddot = coeff_dict["coeff_gammaddot"]
        self.table_friction_factor = coeff_dict["table_friction_factor"]

        ## Get the coefficients that define the cost function for the ball movement in the maze

        self.u_alpha = cost_dict["u_alpha"]
        self.max_control_amp = cost_dict["max_control_amp"]  ## maximum control input amplitude
        self.state_cost_coeff = cost_dict["state_cost_coeff"]  ## array of coefficients for state cost
        self.input_cost_coeff = cost_dict["input_cost_coeff"]  ## array of coefficients for input cost
        self.target_state = cost_dict["target_state"]

        self.return_system_matrix = return_system_matrix

        self.use_PP = use_PP
        self.pp_frac = pp_frac

        self.dX = 6
        self.dU = 2

    def reset(self, reset_state=None):

        if not (reset_state is None):
            self.state = reset_state
        else:
            self.state = np.zeros((self.dX,))

    def forward_dynamics(self, x, u):

        std_X_NP = self.std_X_NP
        std_X_PP = self.std_X_PP

        std_Y = float(self.std_Y)
        alpha_hat = self.alpha_hat
        w_hat = self.w_hat

        h = self.dt

        beta_model_coefficients = self.beta_model_coefficients
        gamma_model_coefficients = self.gamma_model_coefficients

        ##

        x0, x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4], x[5]

        u0, u1 = u[0], u[1]

        ## Get the parametric part
        ## Parametric model---------------------------------------
        phi_acc_rad_visc_model = [
            -(x3**2) * np.sin(x0) * np.cos(x4) ** 2 * np.cos(x0),
            -2 * x3 * x5 * np.cos(x4) * np.cos(x0) ** 2,
            +2 * x3 * x5 * np.cos(x4),
            0.5 * x5**2 * np.sin(2 * x0),
            -9.806 * np.sin(x2) * np.sin(x0),
            -9.806 * np.sin(x4) * np.cos(x2) * np.cos(x0),
            -x1,
        ]

        ## Get the non-parametric model

        f_PP = np.matmul(phi_acc_rad_visc_model, w_hat)

        x_transformed = [x1, x3, x5, np.sin(x0), np.cos(x0), np.sin(x2), np.cos(x2), np.sin(x4), np.cos(x4)] / std_X_NP

        K_test = utils_gp.K_RBF_line(x_transformed, self.X, self.length_scales, self.scaling_factor)
        f_NP = np.matmul(std_Y * (K_test), alpha_hat)[0]  # std_Y*(K_test.T).dot(alpha_hat)

        use_f_PP = 1

        if self.use_PP is False:

            use_f_PP = self.pp_frac
            # use_f_PP =np.random.random_sample()

        f_SP = f_NP + f_PP * use_f_PP

        table_dynamics_variables = [x4, x5, x2, x3, u0, u1]

        betadot = np.matmul(table_dynamics_variables, beta_model_coefficients)
        gammadot = np.matmul(table_dynamics_variables, gamma_model_coefficients)

        c1 = x0 + h * x1  # +dt**2/2*f_SP_model
        c2 = x1 + h * f_SP
        c3 = x2 + h * gammadot
        c4 = gammadot
        c5 = x4 + h * betadot
        c6 = betadot

        self.state = np.array([c1, c2, c3, c4, c5, c6])

        return self.state

    def _step(self, x_u):
        return self.forward_dynamics(x_u[: self.dX], x_u[self.dX :])

    def cost(self, state, u):

        lx = self.state_cost_coeff[0] * (state[0] - self.target_state) ** 2 + self.state_cost_coeff[1] * state[1] ** 2

        u = u / self.max_control_amp
        lu0 = self.input_cost_coeff[0] * self.u_alpha**2 * (np.cosh(u[0] / self.u_alpha) - 1)
        lu1 = self.input_cost_coeff[1] * self.u_alpha**2 * (np.cosh(u[1] / self.u_alpha) - 1)
        lu = lu0 + lu1

        # print lu
        return lx + lu

    def _cost(self, x_u):
        return self.cost(x_u[: self.dX], x_u[self.dX :])

    def rollout(self, actionTraj, T):

        ## Initialize the state here
        ## or initialize the state after instantiating the class

        # self.Maze.state[0] = 2 * np.pi / 4

        StateTraj = []
        # ActionTraj=[]
        CostTraj = []
        StateTraj.append(self.state)

        for t in range(T):

            u = actionTraj[t, 0:2]
            u = np.reshape(u, (self.dU,))

            state_tplusone = self.forward_dynamics(self.state, np.squeeze(u))
            cost = self.cost(self.state, u)
            state_tplusone = np.squeeze(state_tplusone)  # np.reshape(state_tplusone,(self.dX,))

            StateTraj.append(state_tplusone)
            CostTraj.append(cost)

        return StateTraj, actionTraj, CostTraj


if __name__ == "__main__":

    dX = 6
    dU = 2
    # T=60

    ##========================================================

    ## state_cost_coeff: the coefficient with which we weigh the cost for each state component
    ## control_cost_coeff: coefficient with which we weigh the cost for each control component

    state_cost_coeff = np.zeros((dX,))

    state_cost_coeff[0] = 1
    state_cost_coeff[1] = 0.1

    control_cost_coeff = np.zeros((dU,))

    control_cost_coeff[0] = 1
    control_cost_coeff[1] = 1

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

    cost_dict = {
        "u_alpha": 0.2,
        "max_control_amp": 0.2,
        "state_cost_coeff": state_cost_coeff,
        "input_cost_coeff": control_cost_coeff,
        "target_state": np.pi / 2,
    }

    # self.cost_dict=cost_dict ## This is added to the class to get the ring dependent coefficients

    location_model = "./exploration_sines_30Hz/Models/"  # /Envs/Dynamics/CircularMaze

    flg_model = "SP"
    file_name = "Exploration_data_maze_sum_sines_freq1bang5_run2theta_acc_acaus"  #'Exploration_data_maze_sum_sines_freq1bang5theta_acc_acausall'#

    ring = 1

    param_dict = utils_mpc.get_GP_estimator_SP_fromdir(location_model, flg_model, ring, file_name)

    Maze = GaussianProcessMazeModel(param_dict, coeff_dict, cost_dict)

    next_state = Maze.forward_dynamics(np.zeros(dX), np.zeros(dU))
    print("next state ", next_state)
    # Cost_function=Maze_cost_object(cost_dict)
    # Cost_function._get_cost_derivatives()

###=============== Initialization =================================================

# def step_input_profile(max_amp,T,dt,dU):
#
# 	#f=10.
# 	u=np.zeros((T,dU))
# 	f=1.25#np.pi/2
# 	f=2*np.pi*f
# 	for t in range(T):
# 		u[t,0]=0.6*max_amp*np.sign(np.sin(f*t*dt))
# 		u[t,1]=-0.6*max_amp*np.sign(np.sin(f*t*dt))
#
# 	return u
#
#
# def sin_input_profile(max_amp,T,dt,dU):
# 	actionTraj=np.zeros((T,dU))
# 	f=0.75#np.pi/2
# 	omega=2*np.pi*f
#
# 	for t in range(T):
# 		actionTraj[t,0]=max_amp*(np.sin(f*t*dt))
# 		actionTraj[t,1]=-max_amp*(np.sin(f*t*dt))
#
# 		#print(actionTraj[t,:])
#
# 	return actionTraj


# ### call rollout for maze to get the state, input and cost trajectory
# u=step_input_profile(cost_dict["max_control_amp"],ocpi.T,coeff_dict['dt'],dU)
# State_traj, Action_traj,Cost_traj =ocpi.MazeGP.rollout(u,ocpi.T)
#
# obs_trace=np.array(State_traj)
# my_theta = obs_trace[:, 0]
# my_theta_dot = obs_trace[:, 1]
# my_beta = obs_trace[:, 2]
# my_beta_dot = obs_trace[:, 3]
# my_gamma = obs_trace[:, 4]
# my_gamma_dot = obs_trace[:, 5]


# plt.figure()
# plt.subplot(3,2,1)
# plt.plot(my_theta,label = 'theta')
# plt.legend()
# plt.subplot(3,2,2)
# plt.plot(my_theta_dot,label = 'theta vel')
# plt.legend()
# plt.subplot(3,2,3)
# plt.plot(my_beta,label = 'beta')
# plt.legend()
# plt.subplot(3,2,4)
# plt.plot(my_beta_dot,label = 'beta vel')
# plt.legend()
# plt.subplot(3,2,5)
# plt.plot(my_gamma,label = 'gamma')
# plt.legend()
# plt.subplot(3,2,6)
# plt.plot(my_gamma_dot,label = 'gamma vel')
# plt.legend()
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.array(Action_traj)[:,0],label = 'beta action')
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(np.array(Action_traj)[:,1],label = 'gamma action')
# plt.legend()
# plt.figure()
# plt.plot(np.array(Cost_traj)[:], label='Cost')
# plt.legend()
# plt.show()

# print("stateTraj dimension", np.array(State_traj).shape)
