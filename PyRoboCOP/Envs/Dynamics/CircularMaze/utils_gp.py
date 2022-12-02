# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys

import numpy as np
from numpy import sin

sys.path.append("./exploration_sines_30Hz/Models/")

location_model = "./exploration_sines_30Hz/Models/"
# location_model = '/homes/romeres/Projects/ArtificialIntelligence/RobotLearningAndSimulation/RobotEnvironments/Maze/Code/Camera/CleanCodes/exploration_sines_30Hz/Models/'
flg_model = "SP"
file_name = "Exploration_data_maze_sum_sines_freq1bang5_run2theta_acc_acaus"


def get_GP_estimator_SP(
    X_file,
    std_X_NP_file,
    std_X_PP_file,
    std_Y_file,
    w_hat_file,
    alpha_hat_file,
    lengthscales_file,
    scaling_factor_file,
):
    """Function that returns the GP estimation in the SP case,
    when the NP component is given by RBF and the model driven component is
    given by the function 'utils_phi_func.phi_acc_rad_visc_model'.
    input x is composed by:
    [theta,theta_dot,beta,beta_dot,gamma,gamma_dot]
    """
    # load values
    X = np.loadtxt(X_file, unpack=True).T
    std_X_NP = np.loadtxt(std_X_NP_file, unpack=True)
    std_X_PP = np.loadtxt(std_X_PP_file, unpack=True)
    std_Y = np.loadtxt(std_Y_file, unpack=True)
    length_scales = np.loadtxt(lengthscales_file, unpack=True)
    scaling_factor = np.loadtxt(scaling_factor_file, unpack=True)
    w_hat = np.loadtxt(w_hat_file, unpack=True)
    alpha_hat = np.loadtxt(alpha_hat_file, unpack=True)

    param_dict = {
        "X": X,
        "std_X_NP": std_X_NP,
        "std_X_PP": std_X_PP,
        "std_Y": std_Y,
        "length_scales": length_scales,
        "scaling_factor": scaling_factor,
        "w_hat": w_hat,
        "alpha_hat": alpha_hat,
    }

    return param_dict


def get_GP_estimator_SP_fromdir(file, flg_model, ring, file_name):
    """Function that return a semiparametrical GP model by calling the function get_GP_estimator_SP().
    This function is implemented only to be more user friendly and code compact"""
    X_file = file + "data_X_f_acc_" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    std_X_NP_file = file + "std_coef_X_f_acc_NP" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    std_X_PP_file = file + "std_coef_X_f_acc_PP" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    std_Y_file = file + "std_coef_Y_f_acc_" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    w_hat_file = file + "w_hat_" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    alpha_hat_file = file + "alpha_hat_" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    lengthscales_file = file + "lengthscales_f_acc_" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"
    scaling_factor_file = file + "scalingfactor_f_acc_" + flg_model + "_ring" + str(ring) + "_" + file_name + ".txt"

    return get_GP_estimator_SP(
        X_file,
        std_X_NP_file,
        std_X_PP_file,
        std_Y_file,
        w_hat_file,
        alpha_hat_file,
        lengthscales_file,
        scaling_factor_file,
    )


def K_RBF_line(x, X, l, sf=1):
    """Function that returns a line of the kernel matrix in the RBF case"""
    return sf * np.exp(-0.5 * get_dist(X, x, l))


def get_dist(X, x, l):
    """Function that computes the distance between a banch of row vectors
    (in X) and the vector x
    """
    return np.sum(((X - x) / l) ** 2, axis=1).reshape(1, -1)


def sin_input_profile(max_amp, T, dt, dU):
    actionTraj = np.zeros((T, dU))
    f = 0.75  # np.pi/2
    omega = 2 * np.pi * f

    for t in range(T):
        actionTraj[t, 0] = max_amp * (sin(f * T * dt))
        actionTraj[t, 1] = -max_amp * (sin(f * T * dt))

    return actionTraj


def step_input_profile(max_amp, T, dt, dU):

    # f=10.
    actionTraj = np.zeros((T, dU))
    f = 1.25  # np.pi/2
    f = 2 * np.pi * f
    for t in range(T):
        actionTraj[t, 0] = -0.5 * max_amp * np.sign(sin(f * T * dt))
        actionTraj[t, 1] = 0.5 * max_amp * np.sign(sin(f * T * dt))

    return actionTraj
