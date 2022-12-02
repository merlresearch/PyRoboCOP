#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import time
from inspect import ismethod

import adolc as ad
import numpy as np
import Solvers.ocp2nlp.CollisionAvoidanceOCP as caocp
import Solvers.ocp2nlp.collocation as colloc


# Checks is a class implements method
def method_exists(instance, method):
    return hasattr(instance, method) and ismethod(getattr(instance, method))


fl_time = 0  # 1 to print time for computing objective, jacobian and hessian using adolc


class OCP2NLP(object):

    """
    ncolloc:      1,2,3,4,5 - order of collocation
    roots:        "legendre","radau","explicit"
    compl:        0 - posed as x_i y_i <= delta (Raghunathan & Biegler, SIOPT 2005) (default)
                  1 - posed as sum over all complementarity for each time-steps x^Ty <= delta
                  2 - posed as a penalty in the objective
    compladapt:   0 - keep delta fixed (default)
                  1 - keep delta = mu
    compleps:     tolerance for complementarity when compladapt = 0, set to 1.e-4 (default)
    autodiff:     0 - user supplies gradients
                  1 - use ADOLC
    """

    def __init__(self, ocp, ncolloc, roots, compl=0, compladapt=0, compleps=1.0e-4, autodiff=1):
        self.ocp = ocp
        self.ocp_orig = self.ocp

        self.ncolloc = ncolloc
        self.roots = roots
        self.compl = compl
        self.compladapt = compladapt
        self.compleps = compleps
        self.autodiff = autodiff
        self.time_adolc = 0

        """
        Few sanity checks
        """
        if roots in ["legendre", "radau"] and (ncolloc < 0 or ncolloc > 5):
            raise "Parameter ncolloc must be in {1,...,5} when choosing roots = legendre or radau"
        if roots in ["explicit"] and not (ncolloc == 1):
            raise "Parameter ncolloc must be set to 1 when choosing roots = explicit"
        if not (roots == "legendre" or roots == "radau" or roots == "explicit"):
            raise "Parameter roots must be in {legendre,radau,explicit}"
        if not (compl == 0 or compl == 1 or compl == 2):
            raise "Parameter compl must be in {0,1}"
        if not (compladapt == 0 or compladapt == 1):
            raise "Parameter compladapt must be in {0,1}"
        if not (compleps > 0):
            raise "Parameter compleps must be positive"
        if not (autodiff == 0 or autodiff == 1):
            raise "Parameter autodiff must be in {0,1}"

        # get the OCP problem sizes
        (
            self.n_d,
            self.n_a,
            self.n_u,
            self.n_p,
            self.n_cc,
            self.T,
            self.times,
            self.nnz_jac,
            self.nnz_hess,
        ) = self.ocp.get_info()

        if not (self.n_d > 0):
            raise "Number of differential variables must be > 0"
        if not (self.n_u > 0):
            raise "Number of control variables must be > 0"
        if self.n_a < 0:
            raise "Number of algebraic variables must be > 0"
        if self.n_cc < 0:
            raise "Number of algebraic variables must be > 0"
        if not (self.T > 0):
            raise "Number of time steps must be > 0"
        if not (self.T + 1 == len(self.times)):
            raise "Length of times array must be equal to (T+1)"
        if self.n_cc > 0 and not (self.ncolloc == 1):
            raise "Parameter ncolloc must be set to 1 when specifying complementarity constraints"
        if self.n_p < 0:
            raise "Number of parameters must be >= 0"
        if self.n_p > 0:
            if self.autodiff == 0:
                raise "Parameter support not provided for autodiff=0 (ADOLC disabled). Set autodiff=1"
            if self.n_p > 0:
                if not (method_exists(self.ocp, "bounds_params")):
                    raise "When number of parameters > 0, user must implement bounds_params to return lb,ub of parameters"
                if not (method_exists(self.ocp, "initialpoint_params")):
                    raise "When number of parameters > 0, user must implement initialpoint_params to return initial guess for parameters"

        self.time_interval = [(self.times[t + 1] - self.times[t]) for t in range(self.T)]

        if min(self.time_interval) <= 0:
            raise "Length of time intervals must all be > 0"

        """
        self.n_p = 0
        if method_exists(self.ocp,"get_num_params"):
            self.n_p = self.ocp.get_num_params()
            if self.n_p < 0:
                raise "Number of parameters must be >= 0"
            if self.autodiff == 0:
                raise "Parameter support not provided for autodiff=0 (ADOLC disabled). Set autodiff=1"
            if self.n_p > 0:
                if not(method_exists(self.ocp,"get_paramsbounds")):
                    raise "When number of parameters > 0, user must implement get_paramsbounds to return lb,ub of parameters"
                if not(method_exists(self.ocp,"get_paramsinitialcondition")):
                    raise "When number of parameters > 0, user must implement get_paramsinitialcondition to return lb,ub of parameters"
        """

        self.obj_mayer = False
        if method_exists(self.ocp, "objective_mayer"):
            self.obj_mayer = True

        # get the complementarity constraint info
        # cc_var1   - index of complementarity variables
        # cc_bnd1   - 0/1 array 0: lower bound, 1: upper bound
        # cc_var2   - index of complementarity variables
        # cc_bnd2   - 0/1 array 0: lower bound, 1: upper bound
        # complemntarity constraints for i = 1,...,n_cc
        #   if cc_bnd1 = 0, cc_bnd2 = 0
        #       x[cc_var1[i]]-lb[cc_var1[j]] >= 0 \perp x[cc_var2[i]]-lb[cc_var2[i]] >= 0
        #   if cc_bnd2 = 0, cc_bnd2 = 1
        #       x[cc_var1[i]]-lb[cc_var1[j]] >= 0 \perp ub[cc_var2[i]]-x[cc_var2[i]] >= 0
        #   if cc_bnd1 = 1, cc_bnd2 = 0
        #       lb[cc_var1[i]]-x[cc_var1[i]] >= 0 \perp x[cc_var2[i]]-lb[cc_var2[i]] >= 0
        #   if cc_bnd2 = 1, cc_bnd2 = 1
        #       ub[cc_var2[i]]-x[cc_var1[i]] >= 0 \perp ub[cc_var2[i]]-x[cc_var2[i]] >= 0
        if self.n_cc > 0:
            self.cc_var1, self.cc_bnd1, self.cc_var2, self.cc_bnd2 = self.ocp.get_complementarity_info()

            if any(self.cc_var1) < 0 or any(self.cc_var2) < 0:
                raise "The variable indices for complementarity constraints must be > 0 and < number of algebraic variables"
            if not (any(self.cc_bnd1) in range(2)) or not (any(self.cc_bnd2) in range(2)):
                raise "The bound for variables in complementarity constraints must be in {0,1}"

        # check if obstacle avoidance is required, then we wrap the original ocp with another one
        if method_exists(self.ocp, "get_objects_info"):
            if self.autodiff == 0:
                raise "Collision avoidance requires automatic differentiation"
            self.ocp_orig = self.ocp
            self.ocp = caocp.CollisionAvoidanceOCP(self.ocp_orig)
            (
                self.n_d,
                self.n_a,
                self.n_u,
                self.n_p,
                self.n_cc,
                self.T,
                self.times,
                self.nnz_jac,
                self.nnz_hess,
            ) = self.ocp.get_info()
            self.cc_var1, self.cc_bnd1, self.cc_var2, self.cc_bnd2 = self.ocp.get_complementarity_info()

        self.mu = 0.1
        self.compldelta = 1.0  # keeps track of the complementarity relaxation value
        self.dist_con_eps = (
            1.0e-4  # tolerance used in the non-degenerate formulaiton of the collision avoidance constraint
        )

        # get Ipopt problem sizes
        self.n_ipopt = self.T * (self.ncolloc + 1) * self.n_d  # diff vars
        self.n_ipopt = self.n_ipopt + self.n_d  # diff var at time T
        self.n_ipopt = self.n_ipopt + self.T * self.ncolloc * self.n_d  # derivatives of diff vars
        self.n_ipopt = self.n_ipopt + self.T * self.ncolloc * self.n_a  # algebraic vars
        self.n_ipopt = self.n_ipopt + self.T * self.ncolloc * self.n_u  # control vars
        self.n_ipopt = self.n_ipopt + self.n_p  # parameters

        # constraints
        self.m_ipopt = self.T * self.ncolloc * self.n_d  # relate diff vars to derivatives
        self.m_ipopt = self.m_ipopt + self.T * self.n_d  # continuity of diff vars
        self.m_ipopt = self.m_ipopt + self.T * self.ncolloc * self.n_d  # differential eqns
        self.m_ipopt = self.m_ipopt + self.T * self.ncolloc * self.n_a  # algebraic eqns
        if compl == 1:
            self.m_ipopt = self.m_ipopt - self.T * self.ncolloc * self.n_cc + self.T
        if compl == 2:
            self.m_ipopt = self.m_ipopt - self.T * self.ncolloc * self.n_cc

        # get the info on collocation
        self.collocation = colloc.collocation(self.ncolloc, self.roots)

        # quantities that are useful --
        # TODO: change for complementarity formulation
        self.n_var_colloc = 2 * self.n_d + self.n_a + self.n_u  # number vars at a collocation point
        self.n_var_tstep = self.n_d + self.n_var_colloc * self.ncolloc  # number of vars per time step

        self.n_con_colloc = 2 * self.n_d + self.n_a  # number of cons at a collocation point
        self.n_con_tstep = self.n_d + self.n_con_colloc * self.ncolloc  # number of cons per time step
        if self.compl == 1:
            self.n_con_colloc = 2 * self.n_d + self.n_a - self.n_cc
            self.n_con_tstep = self.n_d + self.n_con_colloc * self.ncolloc + 1
        if self.compl == 2:
            self.n_con_colloc = 2 * self.n_d + self.n_a - self.n_cc
            self.n_con_tstep = self.n_d + self.n_con_colloc * self.ncolloc

        # quantities that are useful when user supplies the derivatives
        if self.autodiff == 0:
            # nnz for jacobian
            self.nnz_jac_ipopt = self.T * self.nnz_jac * self.ncolloc  # number of nonzeros in DAE
            self.nnz_jac_ipopt = self.nnz_jac_ipopt + self.T * self.ncolloc * self.n_d * (self.ncolloc + 2)
            # every eqn relating derivative to diff vars
            # depends on all diff vars in the finite element

            # number of nonzeros in the continuity equations
            if self.roots == "legendre":
                self.nnz_jac_ipopt = self.nnz_jac_ipopt + self.T * self.n_d * (self.ncolloc + 2)
            elif self.roots == "radau" or self.roots == "explicit":
                self.nnz_jac_ipopt = self.nnz_jac_ipopt + self.T * self.n_d * 2

            # nonzeros for complementarity constraints
            if not (self.compl == 2):
                self.nnz_jac_ipopt = self.nnz_jac_ipopt + self.T * 2 * self.n_cc * self.ncolloc

            # TODO: change depending on how complementarity variables are represented
            self.nnz_hess_ipopt = self.T * self.nnz_hess * self.ncolloc

            # nonzeros for complementarity constraints
            self.nnz_hess_ipopt = self.nnz_hess_ipopt + self.T * self.n_cc * self.ncolloc

            self.nnz_jac_colloc = self.nnz_jac + self.n_d * (self.ncolloc + 2)
            if not (self.compl == 2):
                self.nnz_jac_colloc = self.nnz_jac_colloc + 2 * self.n_cc
            self.nnz_jac_tstep = self.nnz_jac_colloc * self.ncolloc
            if self.roots == "legendre":
                self.nnz_jac_tstep = self.nnz_jac_tstep + self.n_d * (self.ncolloc + 2)
            elif self.roots == "radau" or self.roots == "explicit":
                self.nnz_jac_tstep = self.nnz_jac_tstep + self.n_d * 2

            self.nnz_hess_colloc = self.nnz_hess
            self.nnz_hess_tstep = self.nnz_hess_colloc * self.ncolloc
            self.nnz_hess_tstep = self.nnz_hess_tstep + self.n_cc * self.ncolloc

            # compute some offset arrays for self.roots == "explicit"
            jac_rowt = np.zeros((self.nnz_jac, 1))
            jac_colt = np.zeros((self.nnz_jac, 1))
            self.ocp.jacobianstructure(jac_rowt, jac_colt)
            self.jac_colt_offset_exp = np.zeros((self.nnz_jac, 1))
            mask = np.where(jac_colt < self.n_d * np.ones((self.nnz_jac, 1), dtype=int))
            self.jac_colt_offset_exp[mask] -= self.n_d

            if self.nnz_hess > 0:
                hess_rowt = np.zeros((self.nnz_hess, 1))
                hess_colt = np.zeros((self.nnz_hess, 1))
                self.ocp.hessianstructure(hess_rowt, hess_colt)
                self.hess_rowt_offset_exp = np.zeros((self.nnz_hess, 1))
                self.hess_colt_offset_exp = np.zeros((self.nnz_hess, 1))
                mask = np.where(hess_rowt < self.n_d * np.ones((self.nnz_hess, 1), dtype=int))
                self.hess_rowt_offset_exp[mask] -= self.n_d
                mask = np.where(hess_colt < self.n_d * np.ones((self.nnz_hess, 1), dtype=int))
                self.hess_colt_offset_exp[mask] -= self.n_d

        # quantities that are useful when using ADOLC
        self.tape_num = 1
        if self.autodiff == 1:
            x0_adolc = np.ones((self.n_ipopt,))
            mu0_adolc = 1.0
            pi0_adolc = np.ones((self.m_ipopt,))
            obj_factor_adolc = 1.0

            # trace objective function
            self.tape_num_obj = self.tape_num
            self.tape_num = self.tape_num + 1
            ad.trace_on(self.tape_num_obj)
            ax = ad.adouble(x0_adolc)
            ad.independent(ax)
            ay = self.objective_(ax)
            ad.dependent(ay)
            ad.trace_off()

            # trace constraint function
            self.tape_num_con = self.tape_num
            self.tape_num = self.tape_num + 1
            ad.trace_on(self.tape_num_con)
            ax = ad.adouble(x0_adolc)
            amu = ad.adouble(mu0_adolc)
            ad.independent(ax)
            ad.independent(amu)
            ay = self.constraints_(ax, amu)
            ad.dependent(ay)
            ad.trace_off()

            # trace lagrangian function
            self.tape_num_lag = self.tape_num
            self.tape_num = self.tape_num + 1
            ad.trace_on(self.tape_num_lag)
            ax = ad.adouble(x0_adolc)
            alagrange = ad.adouble(pi0_adolc)
            aobj_factor = ad.adouble(obj_factor_adolc)
            amu = ad.adouble(mu0_adolc)
            ad.independent(ax)
            ad.independent(alagrange)
            ad.independent(aobj_factor)
            ad.independent(amu)
            ay = self.lagrangian_(ax, alagrange, aobj_factor, amu)
            ad.dependent(ay)
            ad.trace_off()

            options = np.array([1, 1, 0, 0], dtype=int)
            x0_jac = np.hstack([x0_adolc, mu0_adolc])
            result = ad.colpack.sparse_jac_no_repeat(self.tape_num_con, x0_jac, options)

            self.nnz_jac_ipopt_adolc = result[0]
            self.rind_jac_adolc = np.asarray(result[1], dtype=int)
            self.cind_jac_adolc = np.asarray(result[2], dtype=int)
            self.vals_jac_adolc = np.asarray(result[3], dtype=float)

            self.mask_jac_adolc = np.where(self.cind_jac_adolc < self.n_ipopt)

            self.nnz_jac_ipopt = len(self.rind_jac_adolc[self.mask_jac_adolc])

            options = np.array([0, 1], dtype=int)
            x0_hess = np.hstack([x0_adolc, pi0_adolc, obj_factor_adolc, mu0_adolc])

            result = ad.colpack.sparse_hess_no_repeat(self.tape_num_lag, x0_hess, options)

            self.nnz_hess_ipopt_adolc = result[0]
            self.rind_hess_adolc = np.asarray(result[1], dtype=int)
            self.cind_hess_adolc = np.asarray(result[2], dtype=int)
            self.vals_hess_adolc = np.asarray(result[3], dtype=float)

            # need only upper left part of the Hessian
            self.mask_hess_adolc = np.where(self.cind_hess_adolc < self.n_ipopt)

            self.nnz_hess_ipopt = len(self.rind_hess_adolc[self.mask_hess_adolc])

    """
        Method to parse the vector from Ipopt which are arranged as
        (z_0,...,z_{T-1},x_{T,0},p)
        z_t = (x_{t,0},x_{t,1},xdot_{t,1},y_{t,1},u_{t,1},...,x_{t,ncolloc},xdot_{t,ncolloc},y_{t,ncolloc},u_{t,ncolloc})
        x_{t,k}, xdot_{t,k} are n_d-dimensional vectors
        y_{t,k} is n_a-dimensional vector
        u_{t,k} is n_u-dimensional vector
        Returns the vectors for the given t when k is None
        (x_t,xdot_t,y_t,u_t,x_{t+1,0})
        where x_t = [x_{t,0} ... x_{t,ncolloc}], ?_t = [?_{t,1} ... ?_{t,ncolloc}]
    """

    def parse_x_ipopt(self, x_ipopt, t):
        ind_t = t * self.n_var_tstep
        xt = np.ndarray((self.n_d, self.ncolloc + 1))
        if self.autodiff == 1:
            xt = xt.astype(object)
        xdott = np.ndarray((self.n_d, self.ncolloc))
        if self.autodiff == 1:
            xdott = xdott.astype(object)
        yt = np.ndarray((self.n_a, self.ncolloc))
        if self.autodiff == 1:
            yt = yt.astype(object)
        ut = np.ndarray((self.n_u, self.ncolloc))
        if self.autodiff == 1:
            ut = ut.astype(object)
        xt[:, 0] = x_ipopt[ind_t : ind_t + self.n_d]
        ind_t = ind_t + self.n_d
        n_xdxyu = self.n_var_colloc
        for k in range(self.ncolloc):
            xt[:, 1 + k] = x_ipopt[ind_t : ind_t + self.n_d]
            xdott[:, k] = x_ipopt[ind_t + self.n_d : ind_t + 2 * self.n_d]
            yt[:, k] = x_ipopt[ind_t + 2 * self.n_d : ind_t + 2 * self.n_d + self.n_a]
            ut[:, k] = x_ipopt[ind_t + 2 * self.n_d + self.n_a : ind_t + 2 * self.n_d + self.n_a + self.n_u]
            ind_t = ind_t + n_xdxyu
        params = []
        if self.n_p > 0:
            params = np.ndarray((self.n_p, 1))
            if self.autodiff == 1:
                params = params.astype(object)
            for i in range(self.n_p):
                params[-i - 1, 0] = x_ipopt[-i - 1]
        return xt, xdott, yt, ut, np.reshape(x_ipopt[ind_t : ind_t + self.n_d], (self.n_d, 1)), params

    """
        Get the time at the collocation point k in the interval t
    """

    def get_time(self, t, k):
        return self.times[t] + self.time_interval[t] * self.collocation.rx[k]

    def print_ocp(self):
        print("Info on the OCP ---")
        print("number of diff vars   = ", self.n_d)
        print("number of alg  vars   = ", self.n_a)
        print("number of ctrl vars   = ", self.n_u)
        print("number of nnz in jac  = ", self.nnz_jac)
        print("number of nnz in hess = ", self.nnz_hess)

    def bounds(self):
        self.initialcondition()
        lb = np.zeros((self.n_ipopt, 1))
        ub = np.zeros((self.n_ipopt, 1))
        n_xdxyu = self.n_var_colloc
        for t in range(0, self.T):
            lbt, ubt = self.ocp.bounds(t)
            lbt = np.reshape(lbt, (self.n_var_colloc, 1))
            ubt = np.reshape(ubt, (self.n_var_colloc, 1))
            ind_t = t * self.n_var_tstep
            # ... for x_{t,0}
            # when using "radau" or "explicit" we do not need to add bounds for x_{t,0}
            if self.roots == "legendre":
                lb[ind_t : ind_t + self.n_d] = np.reshape(lbt[0 : self.n_d], (self.n_d, 1))
                ub[ind_t : ind_t + self.n_d] = np.reshape(ubt[0 : self.n_d], (self.n_d, 1))
            else:
                lb[ind_t : ind_t + self.n_d] = -np.Infinity * np.ones((self.n_d, 1))
                ub[ind_t : ind_t + self.n_d] = np.Infinity * np.ones((self.n_d, 1))
            ind_t = ind_t + self.n_d
            # for x_{t,k},xdot_{t,k},y_{t,k},u_{t,k}
            lb[ind_t : ind_t + n_xdxyu * self.ncolloc] = np.tile(lbt, (self.ncolloc, 1))
            ub[ind_t : ind_t + n_xdxyu * self.ncolloc] = np.tile(ubt, (self.ncolloc, 1))
        # this is for diff var at time T
        ind_t = self.T * self.n_var_tstep
        lbt, ubt = self.ocp.bounds(self.T)
        lb[ind_t : ind_t + self.n_d] = np.reshape(lbt[0 : self.n_d], (self.n_d, 1))
        ub[ind_t : ind_t + self.n_d] = np.reshape(ubt[0 : self.n_d], (self.n_d, 1))
        # set the initial condition bounds
        lb[: self.n_d] = np.reshape(self.x0, (self.n_d, 1))
        ub[: self.n_d] = np.reshape(self.x0, (self.n_d, 1))
        if method_exists(self.ocp_orig, "bounds_finaltime"):
            lbxf, ubxf = self.ocp_orig.bounds_finaltime()
            for i in range(self.n_d):
                lb[-self.n_p - i - 1] = lbxf[-i - 1]
                ub[-self.n_p - i - 1] = ubxf[-i - 1]

        lbcon = np.zeros((self.m_ipopt, 1))
        ubcon = np.zeros((self.m_ipopt, 1))
        if self.n_cc > 0:
            for t in range(self.T):
                if self.compl == 0:
                    ind_c = t * self.n_con_tstep + (self.n_d + self.n_a - self.n_cc) * self.ncolloc
                    lbcon[ind_c : ind_c + self.n_cc * self.ncolloc] = -1.0e30
                    ubcon[ind_c : ind_c + self.n_cc * self.ncolloc] = 0.0
                elif self.compl == 1:
                    ind_c = t * self.n_con_tstep + (self.n_d + self.n_a - self.n_cc) * self.ncolloc
                    lbcon[ind_c : ind_c + 1] = -1.0e30
                    ubcon[ind_c : ind_c + 1] = 0.0
        # add bounds for parameters
        if self.n_p > 0:
            plb, pub = self.ocp.bounds_params()
            for i in range(self.n_p):
                lb[-i - 1] = plb[-i - 1]
                ub[-i - 1] = pub[-i - 1]
        return lb, ub, lbcon, ubcon

    def initialpoint(self):
        x0_ipopt = np.zeros((self.n_ipopt, 1))
        n_xdxyu = self.n_var_colloc
        for t in range(0, self.T):
            xt, xdott, yt, ut = self.ocp.initialpoint(t)
            ind_t = t * self.n_var_tstep
            # ... for x_{t,0}
            x0_ipopt[ind_t : ind_t + self.n_d] = np.reshape(xt, (self.n_d, 1))
            ind_t = ind_t + self.n_d
            if self.n_a > 0:
                xdxyu = np.row_stack(
                    (
                        np.reshape(xt, (self.n_d, 1)),
                        np.reshape(xdott, (self.n_d, 1)),
                        np.reshape(yt, (self.n_a, 1)),
                        np.reshape(ut, (self.n_u, 1)),
                    )
                )
            else:
                xdxyu = np.row_stack(
                    (np.reshape(xt, (self.n_d, 1)), np.reshape(xdott, (self.n_d, 1)), np.reshape(ut, (self.n_u, 1)))
                )
            # for x_{t,k},xdot_{t,k},y_{t,k},u_{t,k}
            x0_ipopt[ind_t : ind_t + n_xdxyu * self.ncolloc] = np.tile(xdxyu, (self.ncolloc, 1))
        # this is for diff var at time T
        xt, xdott, yt, ut = self.ocp.initialpoint(self.T)
        ind_t = self.T * self.n_var_tstep
        x0_ipopt[ind_t : ind_t + self.n_d] = np.reshape(xt, (self.n_d, 1))
        # add initial guess for parameters
        if self.n_p > 0:
            p0 = self.ocp.initialpoint_params()
            for i in range(self.n_p):
                x0_ipopt[-i - 1] = p0[-i - 1]
        return x0_ipopt

    def initialcondition(self):
        self.x0 = self.ocp.initialcondition()

    def parsesolution(self, x_ipopt):
        x_sol = np.ndarray([self.T + 1, self.n_d, self.ncolloc + 1])
        xdot_sol = np.ndarray([self.T, self.n_d, self.ncolloc])
        y_sol = np.ndarray([self.T, self.n_a, self.ncolloc])
        u_sol = np.ndarray([self.T, self.n_u, self.ncolloc])
        for t in range(0, self.T):
            xt, xdott, yt, ut, xt1, p_sol = self.parse_x_ipopt(x_ipopt, t)
            x_sol[t, :, :] = xt
            xdot_sol[t, :, :] = xdott
            y_sol[t, :, :] = yt
            u_sol[t, :, :] = ut
        x_sol[self.T, :] = xt1
        if self.n_p > 0:
            return x_sol, xdot_sol, y_sol, u_sol, p_sol
        return x_sol, xdot_sol, y_sol, u_sol

    def get_complementarity_objective(self, t, y, lby, uby):
        obj_cc = 0.0
        compldelta = self.get_complementarity_relaxation()
        for i in range(self.n_cc):
            consi = 1.0
            if self.cc_bnd1[i] == 0:
                consi = consi * (y[self.cc_var1[i]] - lby[self.cc_var1[i]])
            else:
                consi = consi * (uby[self.cc_var1[i]] - y[self.cc_var1[i]])
            if self.cc_bnd2[i] == 0:
                consi = consi * (y[self.cc_var2[i]] - lby[self.cc_var2[i]])
            else:
                consi = consi * (uby[self.cc_var2[i]] - y[self.cc_var2[i]])
            obj_cc = obj_cc + consi
        obj_cc = obj_cc / compldelta
        return obj_cc

    def get_complementarity_objective_gradient(self, grad_cc, t, y, lby, uby):
        grad_cc[:, 0] = 0.0
        compldelta = self.get_complementarity_relaxation()
        for i in range(self.n_cc):
            if self.cc_bnd1[i] == 0 and self.cc_bnd2[i] == 0:
                grad_cc[self.cc_var1[i]] = (y[self.cc_var2[i]] - lby[self.cc_var2[i]]) / compldelta
                grad_cc[self.cc_var2[i]] = (y[self.cc_var1[i]] - lby[self.cc_var1[i]]) / compldelta
            elif self.cc_bnd1[i] == 0 and self.cc_bnd2[i] == 1:
                grad_cc[self.cc_var1[i]] = (uby[self.cc_var2[i]] - y[self.cc_var2[i]]) / compldelta
                grad_cc[self.cc_var2[i]] = -(y[self.cc_var1[i]] - lby[self.cc_var1[i]]) / compldelta
            elif self.cc_bnd1[i] == 1 and self.cc_bnd2[i] == 0:
                grad_cc[self.cc_var1[i]] = -(y[self.cc_var2[i]] - lby[self.cc_var2[i]]) / compldelta
                grad_cc[self.cc_var2[i]] = (uby[self.cc_var1[i]] - y[self.cc_var1[i]]) / compldelta
            else:
                grad_cc[self.cc_var1[i]] = -(uby[self.cc_var2[i]] - y[self.cc_var2[i]]) / compldelta
                grad_cc[self.cc_var2[i]] = -(uby[self.cc_var1[i]] - y[self.cc_var1[i]]) / compldelta

    def get_complementarity_constraints(self, cons_cc, t, y, lby, uby, complrhs):
        for i in range(self.n_cc):
            consi = 1.0
            if self.cc_bnd1[i] == 0:
                consi = consi * (y[self.cc_var1[i]] - lby[self.cc_var1[i]])
            else:
                consi = consi * (uby[self.cc_var1[i]] - y[self.cc_var1[i]])
            if self.cc_bnd2[i] == 0:
                consi = consi * (y[self.cc_var2[i]] - lby[self.cc_var2[i]])
            else:
                consi = consi * (uby[self.cc_var2[i]] - y[self.cc_var2[i]])
            cons_cc[i] = consi - complrhs  # self.get_complementarity_relaxation()

    def get_complementarity_relaxation(self):
        if self.compl == 0:
            alpha = 1.0
        elif self.compl == 1:
            alpha = self.n_cc
        elif self.compl == 2:
            alpha = 1.0
        if self.compladapt == 0:
            self.compldelta = alpha * self.compleps
        elif self.compladapt == 1:
            self.compldelta = alpha * max(1.0e-6, min(self.compldelta, self.mu))
        return self.compldelta

    def get_complementarity_jacobian(self, jac, t, y, lby, uby, complrhs):
        for i in range(self.n_cc):
            if self.cc_bnd1[i] == 0 and self.cc_bnd2[i] == 0:
                jac[i] = y[self.cc_var2[i]] - lby[self.cc_var2[i]]
                jac[self.n_cc + i] = y[self.cc_var1[i]] - lby[self.cc_var1[i]]
            elif self.cc_bnd1[i] == 1 and self.cc_bnd2[i] == 0:
                jac[i] = -(y[self.cc_var2[i]] - lby[self.cc_var2[i]])
                jac[self.n_cc + i] = uby[self.cc_var1[i]] - y[self.cc_var1[i]]
            elif self.cc_bnd1[i] == 0 and self.cc_bnd2[i] == 1:
                jac[i] = uby[self.cc_var2[i]] - y[self.cc_var2[i]]
                jac[self.n_cc + i] = -(y[self.cc_var1[i]] - lby[self.cc_var1[i]])
            else:
                jac[i] = -(uby[self.cc_var2[i]] - y[self.cc_var2[i]])
                jac[self.n_cc + 1] = -(uby[self.cc_var1[i]] - y[self.cc_var1[i]])

    def get_complementarity_hessian(self, hess, t, y, mult_cc, complrhs):
        compldelta = self.get_complementarity_relaxation()
        for i in range(self.n_cc):
            if self.cc_bnd1[i] == self.cc_bnd2[i]:
                if self.compl == 0:
                    hess[i] = mult_cc[i]
                elif self.compl == 1:
                    hess[i] = mult_cc[0]
                elif self.compl == 2:
                    hess[i] = mult_cc[0] / compldelta
            else:
                if self.compl == 0:
                    hess[i] = -mult_cc[i]
                elif self.compl == 1:
                    hess[i] = mult_cc[0]
                elif self.compl == 2:
                    hess[i] = mult_cc[0] / compldelta

    """
        traj - 3-d array (# tsteps) x (var dim) x (#collocation points)
        index - specifies variable index (second index)
        diff - specifies if a differential variable is being plotted
    """

    def extract_trajectory(self, traj, index, diff=False):
        tarray = []
        if diff:
            if self.roots == "legendre":
                xarray = np.zeros(self.T * (self.ncolloc + 1) + 1)
                tarray = [self.get_time(t, k) for t in range(self.T) for k in range(self.ncolloc + 1)]
                tarray.append(self.times[self.T])
                for t in range(self.T):
                    xarray[t * (self.ncolloc + 1) : (t + 1) * (self.ncolloc + 1)] = traj[t, index, :]
                xarray[self.T * (self.ncolloc + 1)] = traj[self.T, index, 0]
            else:
                xarray = np.zeros(self.T * (self.ncolloc) + 1)
                tarray = [self.get_time(t, k) for t in range(self.T) for k in range(self.ncolloc)]
                tarray.insert(0, self.get_time(0, 0))
                xarray[0] = traj[0, index, 0]
                for t in range(self.T):
                    xarray[1 + t * self.ncolloc : 1 + (t + 1) * self.ncolloc] = traj[t, index, 1:]
        else:
            xarray = np.zeros(self.T * self.ncolloc)
            tarray = [self.get_time(t, k + 1) for t in range(self.T) for k in range(self.ncolloc)]
            for t in range(self.T):
                xarray[t * self.ncolloc : (t + 1) * self.ncolloc] = traj[t, index, :]
        return tarray, xarray

    """
        ********************************************************
        Implementations of methods that will be called by Ipopt.
        ********************************************************
    """

    """
        Objective of the OCP
    """

    def objective(self, x_ipopt):
        st_time = time.time()
        if self.autodiff == 0:
            return self.objective_(x_ipopt)
        else:
            obj = ad.function(self.tape_num_obj, x_ipopt)[0]
            if fl_time:
                print("Time objective: ", time.time() - st_time)
            self.time_adolc = self.time_adolc + time.time() - st_time
            return obj

    def objective_(self, x_ipopt):
        obj = 0.0
        params = []
        for t in range(0, self.T):
            xt, xdott, yt, ut, xt1, params = self.parse_x_ipopt(x_ipopt, t)
            lbt, ubt = self.ocp.bounds(t)
            for k in range(self.ncolloc):
                coeffk = self.collocation.lagpolyobjint1[k]
                tk = self.get_time(t, k + 1)
                ht = self.time_interval[t]
                if self.roots == "explicit":
                    obj = obj + ht * self.ocp.objective(tk, xt[:, k], xdott[:, k], yt[:, k], ut[:, k], params)
                else:
                    obj = obj + ht * coeffk * self.ocp.objective(
                        tk, xt[:, k + 1], xdott[:, k], yt[:, k], ut[:, k], params
                    )
                if self.compl == 2:
                    lbyt = lbt[2 * self.n_d : 2 * self.n_d + self.n_a]
                    ubyt = ubt[2 * self.n_d : 2 * self.n_d + self.n_a]
                    obj = obj + ht * self.get_complementarity_objective(tk, yt[:, k], lbyt, ubyt)
        # contribution from the Mayer objective
        if self.obj_mayer:
            x0, xdott, yt, ut, xt1, params = self.parse_x_ipopt(x_ipopt, 0)
            xt, xdott, yt, ut, xf, params = self.parse_x_ipopt(x_ipopt, self.T - 1)
            obj_mayer = self.ocp.objective_mayer(x0, xf, params)
            obj = obj + obj_mayer
        return obj

    """
        Gradient of the Objective of the OCP
    """

    def gradient(self, x_ipopt):
        st_time = time.time()
        if self.autodiff == 0:
            return self.gradient_(x_ipopt)
        else:
            grad = ad.gradient(self.tape_num_obj, x_ipopt)
            if fl_time:
                print("Time gradient objective: ", time.time() - st_time)
            self.time_adolc = self.time_adolc + time.time() - st_time
            return grad

    def gradient_(self, x_ipopt):
        grad = np.zeros((self.n_ipopt, 1))
        n_xdxyu = self.n_var_colloc
        gradtk = np.zeros((n_xdxyu + self.n_p, 1))
        grad_cc = np.zeros((self.n_a, 1))
        for t in range(0, self.T):
            xt, xdott, yt, ut, xt1, params = self.parse_x_ipopt(x_ipopt, t)
            ind_t = t * self.n_var_tstep + self.n_d
            lbt, ubt = self.ocp.bounds(t)
            for k in range(self.ncolloc):
                coeffk = self.collocation.lagpolyobjint1[k]
                tk = self.get_time(t, k + 1)
                ht = self.time_interval[t]
                if self.roots == "explicit":
                    self.ocp.gradient(gradtk, tk, xt[:, k], xdott[:, k], yt[:, k], ut[:, k], params)
                    grad[ind_t - self.n_d : ind_t] = ht * gradtk[: self.n_d]
                    grad[ind_t + self.n_d : ind_t + n_xdxyu] = ht * gradtk[self.n_d : n_xdxyu]
                else:
                    self.ocp.gradient(gradtk, tk, xt[:, k + 1], xdott[:, k], yt[:, k], ut[:, k], params)
                    grad[ind_t : ind_t + n_xdxyu] = ht * coeffk * gradtk[:n_xdxyu]
                if self.n_cc > 0 and self.compl == 2:
                    lbyt = lbt[2 * self.n_d : 2 * self.n_d + self.n_a]
                    ubyt = ubt[2 * self.n_d : 2 * self.n_d + self.n_a]
                    self.get_complementarity_objective_gradient(grad_cc, tk, yt[:, k], lbyt, ubyt)
                    grad[ind_t + 2 * self.n_d : ind_t + 2 * self.n_d + self.n_a] += ht * grad_cc
                if self.n_p > 0:
                    grad[self.n_ipopt - self.n_p :] = ht * coeffk * gradtk[n_xdxyu:]
                ind_t = ind_t + n_xdxyu
        return grad

    """
        Constraints of the OCP
    """

    def constraints(self, x_ipopt):
        complrhs = self.get_complementarity_relaxation()
        st_time = time.time()
        if self.autodiff == 0:
            cons = self.constraints_(x_ipopt, complrhs)
        else:
            x = np.hstack([x_ipopt, complrhs])
            cons = ad.function(self.tape_num_con, x)
            if fl_time:
                print("Time constraints: ", time.time() - st_time)
            self.time_adolc = self.time_adolc + time.time() - st_time
        return cons

    def constraints_(self, x_ipopt, complrhs):
        cons = np.zeros((self.m_ipopt, 1))
        constk = np.zeros((self.n_d + self.n_a - self.n_cc, 1))
        cons_cc = np.zeros((self.n_cc, 1))
        if self.autodiff == 1:
            cons = cons.astype(object)
            constk = constk.astype(object)
            cons_cc = cons_cc.astype(object)
        for t in range(0, self.T):
            ind_c = t * self.n_con_tstep
            xt, xdott, yt, ut, xt1, params = self.parse_x_ipopt(x_ipopt, t)
            lbt, ubt = self.ocp.bounds(t)
            # DAE
            for k in range(self.ncolloc):
                tk = self.get_time(t, k + 1)
                if self.roots == "explicit":
                    self.ocp.constraint(constk, tk, xt[:, k], xdott[:, k], yt[:, k], ut[:, k], params)
                else:
                    self.ocp.constraint(constk, tk, xt[:, k + 1], xdott[:, k], yt[:, k], ut[:, k], params)
                cons[ind_c : ind_c + self.n_d + self.n_a - self.n_cc] = constk
                ind_c = ind_c + self.n_d + self.n_a - self.n_cc
            # complementarity constraints
            if self.n_cc > 0 and self.compl < 2:
                lbyt = lbt[2 * self.n_d : 2 * self.n_d + self.n_a]
                ubyt = ubt[2 * self.n_d : 2 * self.n_d + self.n_a]
                if self.compl == 0:
                    for k in range(self.ncolloc):
                        self.get_complementarity_constraints(cons_cc, t, yt[:, k], lbyt, ubyt, complrhs)
                        cons[ind_c : ind_c + self.n_cc] = cons_cc
                        ind_c = ind_c + self.n_cc
                elif self.compl == 1:
                    for k in range(self.ncolloc):
                        self.get_complementarity_constraints(cons_cc, t, yt[:, k], lbyt, ubyt, complrhs)
                        cons[ind_c] = cons[ind_c] + sum(cons_cc)
                    ind_c = ind_c + 1
            # relate derivatives to diff vars
            Mlagmat = self.collocation.lagpolyxder[:, 1:]
            xdottmat = np.matmul(xt, Mlagmat)
            ht = self.time_interval[t]
            cons[ind_c : ind_c + self.n_d * self.ncolloc] = np.reshape(
                -xdott + (1.0 / ht) * xdottmat, (self.n_d * self.ncolloc, 1), "F"
            )
            #            cons[ind_c:ind_c+self.n_d*self.ncolloc,:] = -xdott+(1.0/ht)*xdottmat
            ind_c = ind_c + self.n_d * self.ncolloc
            # continuity equation
            if self.roots == "legendre":
                lagpolyx1 = np.reshape(self.collocation.lagpolyx1, (self.ncolloc + 1, 1))
                cons[ind_c : ind_c + self.n_d] = -xt1 + np.matmul(xt, lagpolyx1)
            elif self.roots == "radau" or self.roots == "explicit":
                cons[ind_c : ind_c + self.n_d] = -xt1 + np.reshape(xt[:, self.ncolloc], (self.n_d, 1))
        return cons

    """
        Sparsity structure of the Jacobian of the OCP
    """

    def jacobianstructure(self):
        if self.autodiff == 0:
            row, col = self.jacobianstructure_()
        else:
            row = self.rind_jac_adolc[self.mask_jac_adolc]
            col = self.cind_jac_adolc[self.mask_jac_adolc]
        return row, col

    def jacobianstructure_(self):
        if self.nnz_jac_ipopt == 0:
            return [], []
        row = np.zeros((self.nnz_jac_ipopt, 1), dtype=int)
        col = np.zeros((self.nnz_jac_ipopt, 1), dtype=int)
        rowt = np.zeros((self.nnz_jac, 1), dtype=int)
        colt = np.zeros((self.nnz_jac, 1), dtype=int)
        self.ocp.jacobianstructure(rowt, colt)
        if self.roots == "explicit":
            colt = colt + self.jac_colt_offset_exp
        n_xdxyu = self.n_var_colloc
        if self.n_cc > 0:
            cc_var12 = np.row_stack((self.cc_var1, self.cc_var2))
        for t in range(0, self.T):
            ind_t = t * self.n_var_tstep + self.n_d
            ind_c = t * self.n_con_tstep
            ind_j = t * self.nnz_jac_tstep
            # DAE
            for k in range(self.ncolloc):
                row[ind_j : ind_j + self.nnz_jac : 1] = rowt + ind_c
                col[ind_j : ind_j + self.nnz_jac : 1] = colt + ind_t
                ind_c = ind_c + self.n_d + self.n_a - self.n_cc
                ind_t = ind_t + n_xdxyu
                ind_j = ind_j + self.nnz_jac
            # complementarity constraints
            if self.n_cc > 0 and self.compl < 2:
                ind_t = t * self.n_var_tstep + self.n_d
                if self.compl == 0:
                    for k in range(self.ncolloc):
                        cc_inds = np.reshape(list(range(self.n_cc)), (self.n_cc, 1))
                        row[ind_j : ind_j + 2 * self.n_cc] = ind_c + np.tile(cc_inds, (2, 1))
                        col[ind_j : ind_j + 2 * self.n_cc] = ind_t + 2 * self.n_d + cc_var12
                        ind_c = ind_c + self.n_cc
                        ind_t = ind_t + n_xdxyu
                        ind_j = ind_j + 2 * self.n_cc
                elif self.compl == 1:
                    for k in range(self.ncolloc):
                        row[ind_j : ind_j + 2 * self.n_cc] = ind_c
                        col[ind_j : ind_j + 2 * self.n_cc] = ind_t + 2 * self.n_d + cc_var12
                        ind_t = ind_t + n_xdxyu
                        ind_j = ind_j + 2 * self.n_cc
                    ind_c = ind_c + 1
            # relate derivatives to diff vars
            ind_t = t * self.n_var_tstep + self.n_d
            for k in range(self.ncolloc):
                for d in range(self.n_d):
                    # xdot var
                    row[ind_j] = ind_c
                    col[ind_j] = ind_t + k * n_xdxyu + self.n_d + d
                    ind_j = ind_j + 1
                    # for the right hand sides of the xdot eqn - all x_{t,k}'s
                    row[ind_j : ind_j + self.n_con_colloc + 1] = ind_c
                    col[ind_j] = ind_t - self.n_d + d
                    col[ind_j + 1 : ind_j + self.ncolloc + 1, :] = list(
                        range(ind_t + d, ind_t + self.ncolloc * n_xdxyu + d, n_xdxyu)
                    )
                    ind_j = ind_j + self.ncolloc + 1
                    ind_c = ind_c + 1
            # continuity equations
            for d in range(self.n_d):
                if self.roots == "legendre":
                    row[ind_j] = ind_c
                    col[ind_j] = (t + 1) * self.n_var_tstep + d
                    ind_j = ind_j + 1
                    row[ind_j] = ind_c
                    col[ind_j] = ind_t - self.n_d + d
                    ind_j = ind_j + 1
                    row[ind_j : ind_j + self.ncolloc] = ind_c
                    col[ind_j : ind_j + self.ncolloc, :] = ind_t + n_xdxyu * list(range(self.ncolloc)) + d
                    ind_j = ind_j + self.ncolloc
                    ind_c = ind_c + 1
                elif self.roots == "radau" or self.roots == "explicit":
                    row[ind_j] = ind_c
                    col[ind_j] = (t + 1) * self.n_var_tstep + d
                    ind_j = ind_j + 1
                    row[ind_j] = ind_c
                    col[ind_j] = ind_t + (self.ncolloc - 1) * n_xdxyu + d
                    ind_j = ind_j + 1
                    ind_c = ind_c + 1
        return row, col

    """
        Jacobian of the OCP
    """

    def jacobian(self, x_ipopt):
        st_time = time.time()
        complrhs = self.get_complementarity_relaxation()
        if self.autodiff == 0:
            jac = self.jacobian_(x_ipopt, complrhs)
        else:
            x = np.hstack([x_ipopt, complrhs])
            result = ad.colpack.sparse_jac_repeat(
                self.tape_num_con,
                x,
                self.nnz_jac_ipopt_adolc,
                self.rind_jac_adolc,
                self.cind_jac_adolc,
                self.vals_jac_adolc,
            )

            jac = result[3][self.mask_jac_adolc]

            if fl_time:
                print("Time jacobian: ", time.time() - st_time)
            self.time_adolc = self.time_adolc + time.time() - st_time
        return jac

    def jacobian_(self, x_ipopt, complrhs):
        if self.nnz_jac_ipopt == 0:
            return []
        jac = np.zeros((self.nnz_jac_ipopt, 1))
        jactk = np.zeros((self.nnz_jac, 1))
        if self.n_cc > 0 and self.compl < 2:
            jac_cc = np.zeros((2 * self.n_cc, 1))
        for t in range(0, self.T):
            xt, xdott, yt, ut, xt1, params = self.parse_x_ipopt(x_ipopt, t)
            lbt, ubt = self.ocp.bounds(t)
            lbyt = lbt[2 * self.n_d : 2 * self.n_d + self.n_a]
            ubyt = ubt[2 * self.n_d : 2 * self.n_d + self.n_a]
            ind_j = t * self.nnz_jac_tstep
            # DAE
            for k in range(self.ncolloc):
                tk = self.get_time(t, k)
                if self.roots == "explicit":
                    self.ocp.jacobian(jactk, tk, xt[:, k], xdott[:, k], yt[:, k], ut[:, k], params)
                else:
                    self.ocp.jacobian(jactk, tk, xt[:, k + 1], xdott[:, k], yt[:, k], ut[:, k], params)
                jac[ind_j : ind_j + self.nnz_jac] = jactk
                ind_j = ind_j + self.nnz_jac
            # complementarity constraints
            if self.n_cc > 0 and self.compl < 2:
                for k in range(self.ncolloc):
                    self.get_complementarity_jacobian(jac_cc, t, yt[:, k], lbyt, ubyt, complrhs)
                    jac[ind_j : ind_j + 2 * self.n_cc] = jac_cc
                    ind_j = ind_j + 2 * self.n_cc
            # relate derivatives to diff vars
            Mlagmat = (1.0 / self.time_interval[t]) * self.collocation.lagpolyxder[:, 1:]
            for k in range(self.ncolloc):
                coeff = np.zeros((self.ncolloc + 2, 1))
                coeff[0] = -1.0
                coeff[1:, 0] = Mlagmat[:, k]
                jac[ind_j : ind_j + self.n_d * (self.ncolloc + 2)] = np.tile(coeff, (self.n_d, 1))
                ind_j = ind_j + self.n_d * (self.ncolloc + 2)
            # continuity equations
            for d in range(self.n_d):
                if self.roots == "legendre":
                    coeff = np.zeros((self.ncolloc + 2, 1))
                    coeff[0] = -1.0
                    coeff[1:, 0] = self.collocation.lagpolyx1
                    jac[ind_j : ind_j + self.ncolloc + 2] = coeff
                    ind_j = ind_j + self.ncolloc + 2
                elif self.roots == "radau" or self.roots == "explicit":
                    jac[ind_j : ind_j + 2] = np.reshape([-1.0, 1.0], (2, 1))
                    ind_j = ind_j + 2

        return jac

    """
        Sparsity structure of the Hessian of the Lagrangian of the OCP
    """

    def hessianstructure(self):
        # return 0
        if self.autodiff == 0:
            if self.nnz_hess > 0:
                return self.hessianstructure_()
            else:
                return [], []
        else:
            return self.rind_hess_adolc[self.mask_hess_adolc], self.cind_hess_adolc[self.mask_hess_adolc]

    def hessianstructure_(self):
        # return np.array([0]),np.array([0])
        if self.nnz_hess_ipopt == 0:
            return [], []
        row = np.zeros((self.nnz_hess_ipopt, 1), dtype=int)
        col = np.zeros((self.nnz_hess_ipopt, 1), dtype=int)
        rowt = np.zeros((self.nnz_hess, 1), dtype=int)
        colt = np.zeros((self.nnz_hess, 1), dtype=int)
        self.ocp.hessianstructure(rowt, colt)
        if self.roots == "explicit":
            rowt = rowt + self.hess_rowt_offset_exp
            colt = colt + self.hess_colt_offset_exp
        n_xdxyu = self.n_var_colloc
        for t in range(0, self.T):
            ind_t = t * self.n_var_tstep
            ind_h = t * self.nnz_hess_tstep
            for k in range(self.ncolloc):
                row[ind_h : ind_h + self.nnz_hess : 1] = rowt + ind_t + self.n_d + k * n_xdxyu
                col[ind_h : ind_h + self.nnz_hess : 1] = colt + ind_t + self.n_d + k * n_xdxyu
                ind_h = ind_h + self.nnz_hess
            # for complementarity
            if self.n_cc > 0:
                for k in range(self.ncolloc):
                    row[ind_h : ind_h + self.n_cc] = ind_t + self.n_d + k * n_xdxyu + 2 * self.n_d + self.cc_var1
                    col[ind_h : ind_h + self.n_cc] = ind_t + self.n_d + k * n_xdxyu + 2 * self.n_d + self.cc_var2
                    ind_h = ind_h + self.n_cc
        return row, col

    """
        Hessian of the Lagrangian of the OCP
    """

    def hessian(self, x_ipopt, mult, obj_factor):
        # return 0
        st_time = time.time()
        complrhs = self.get_complementarity_relaxation()
        if self.autodiff == 0:
            return self.hessian_(x_ipopt, mult, obj_factor, complrhs)
        else:
            x = np.hstack([x_ipopt, mult, obj_factor, complrhs])
            result = ad.colpack.sparse_hess_repeat(
                self.tape_num_lag, x, self.rind_hess_adolc, self.cind_hess_adolc, self.vals_hess_adolc
            )
            if fl_time:
                print("Time hessian: ", time.time() - st_time)
            self.time_adolc = self.time_adolc + time.time() - st_time
            return result[3][self.mask_hess_adolc]

    def hessian_(self, x_ipopt, mult, obj_factor, complrhs):
        # return 0
        hess = np.zeros((self.nnz_hess_ipopt, 1))
        hesstk = np.zeros((self.nnz_hess, 1))
        hess_cc = np.zeros((self.n_cc, 1))
        for t in range(0, self.T):
            xt, xdott, yt, ut, xt1, params = self.parse_x_ipopt(x_ipopt, t)
            ind_h = t * self.nnz_hess_tstep
            for k in range(self.ncolloc):
                coeffk = self.collocation.lagpolyobjint1[k]
                tk = self.get_time(t, k)
                ind_c = t * self.n_con_tstep + k * (self.n_d + self.n_a - self.n_cc)
                multt = mult[ind_c : ind_c + self.n_d + self.n_a - self.n_cc]
                obj_factork = obj_factor * self.time_interval[t] * coeffk
                if self.roots == "explicit":
                    self.ocp.hessian(hesstk, tk, xt[:, k], xdott[:, k], yt[:, k], ut[:, k], params, multt, obj_factork)
                else:
                    self.ocp.hessian(
                        hesstk, tk, xt[:, k + 1], xdott[:, k], yt[:, k], ut[:, k], params, multt, obj_factork
                    )
                hess[ind_h : ind_h + self.nnz_hess : 1] = hesstk
                ind_h = ind_h + self.nnz_hess
            # for complementarity
            if self.n_cc > 0:
                for k in range(self.ncolloc):
                    if self.compl == 0:
                        ind_c = t * self.n_con_tstep + self.ncolloc * (self.n_d + self.n_a - self.n_cc) + k * self.n_cc
                        multt = mult[ind_c : ind_c + self.n_cc]
                    elif self.compl == 1:
                        ind_c = t * self.n_con_tstep + self.ncolloc * (self.n_d + self.n_a - self.n_cc)
                        multt = mult[ind_c : ind_c + 1]
                    elif self.compl == 2:
                        ind_c = t * self.n_con_tstep + self.ncolloc * (self.n_d + self.n_a - self.n_cc)
                        multt = obj_factor * self.time_interval[t] * np.array([1.0])
                    self.get_complementarity_hessian(hess_cc, tk, yt[:, k], multt, complrhs)
                    hess[ind_h : ind_h + self.n_cc] = hess_cc
        return hess

    """
        Callback from Ipopt
    """

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        self.mu = mu

    """
        Lagrangian
    """

    def lagrangian_(self, x_ipopt, mult, obj_factor, complrhs):
        return self.objective_(x_ipopt) * obj_factor + np.dot(mult, self.constraints_(x_ipopt, complrhs))
