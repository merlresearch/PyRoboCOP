#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools as itt

import numpy as np

"""
    Computes the necessary collocations coefficients that are used in the discretization.
    Given collocation order ncolloc:
    lagpolyx_coeff          -   (ncolloc+1) x (ncolloc+1) array containing coefficients of the lagrange polynomials
                                the polynomial at root k in {0,...,ncolloc} is:
                                    sum_{p in {0,...,ncolloc}} lagpolyx_coeff[k,p]*tau^(ncolloc-p)
    lagpolyx_fun(k,tau)     -   lambda function to evaluate the lagrange polynomial at root k at a the point tau
    lagpolyxder_fun(k,tau)  -   lambda function to evaluate the derivative of the lagrange polynomial at root k at a the point tau
    lagpolyx1               -   (ncolloc+1) array containing the evaluation of the lagrange polynomials at tau = 1.0
    lagpolyxder             -   (ncolloc+1) x (ncolloc+1) array
                                lagpolyxder[k,j] is the value of the derivative of the lagrange polynomials k evaluated at root j
    lagpolyobj_coeff        -   ncolloc x ncolloc array containing coefficients of the lagrange polynomials
                                the polynomial at root k in {1,...,ncolloc} is:
                                    sum_{p in {0,...,ncolloc-1}} lagpolyobj_coeff[k,p]*tau^(ncolloc-1-p)
    lagpolyobjint_fun(k,tau)-   lambda function to evaluate the integral of the lagrange polynomial k at the point tau
    lagpolyobjint1          -   ncolloc array containing the evaluation of the integral of lagrange polynomials at 1.0
"""


class collocation(object):
    def __init__(self, ncolloc, roots):

        self.ncolloc = ncolloc

        self.roots_legendre = [
            [0.0, 0.5],
            [0.0, 0.211325, 0.788675],
            [0.0, 0.112702, 0.5, 0.887298],
            [0.0, 0.069432, 0.330009, 0.669991, 0.930568],
            [0.0, 0.046910, 0.230765, 0.5, 0.769235, 0.953090],
        ]

        self.roots_radau = [
            [0.0, 1.0],
            [0.0, 0.3333333, 1.0],
            [0.0, 0.155051, 0.6444949, 1.0],
            [0.0, 0.088588, 0.409467, 0.787659, 1.0],
            [0.0, 0.057104, 0.276843, 0.583590, 0.860240, 1.0],
        ]

        if roots == "legendre":
            rx = self.roots_legendre[ncolloc - 1]
        elif roots == "radau":
            rx = self.roots_radau[ncolloc - 1]
        elif roots == "explicit":
            self.rx = [0.0, 0.0]
            self.lagpolyxder = np.array([[0.0, -1.0], [0.0, 1.0]])
            self.lagpolyx1 = [0.0, 1.0]
            self.lagpolyobjint1 = [1.0]
            return

        self.rx = rx

        # coefficients of the lagrange polynomial for diff vars
        self.lagpolyx_coeff = self.lagpoly_coeff(ncolloc, rx)

        # lambda function for evaluating the polynomial
        self.lagpolyx_fun = lambda k, tau: np.sum(
            [self.lagpolyx_coeff[k, p] * pow(tau, ncolloc - p) for p in range(ncolloc + 1)]
        )

        # lambda function for evaluating the derivative of polynomial
        self.lagpolyxder_fun = lambda k, tau: np.sum(
            [(ncolloc - p) * self.lagpolyx_coeff[k, p] * pow(tau, ncolloc - p - 1) for p in range(ncolloc)]
        )

        # we only require the derivative of these polynomials evaluated at the different roots
        # this is needed for relating the time derivatives of diff vars to the vcalues of diff vars
        # {0,...,ncolloc+1} x {0,...,ncolloc+1} array - kth (first index) lagrange polynomial derivative
        # evaluated at the jth (second index) root
        self.lagpolyxder = np.ndarray((ncolloc + 1, ncolloc + 1))
        for k in range(ncolloc + 1):
            self.lagpolyxder[k, :] = [self.lagpolyxder_fun(k, rx[j]) for j in range(ncolloc + 1)]

        # evaluate the collocation polynomials at 1.0 this is used for continuity of diff vars
        self.lagpolyx1 = [self.lagpolyx_fun(k, 1.0) for k in range(ncolloc + 1)]

        # get coefficients for lagrange polynomial for the objective
        self.lagpolyobj_coeff = []
        if ncolloc > 1:
            self.lagpolyobj_coeff = self.lagpoly_coeff(ncolloc - 1, rx[1:])

        # evaluate the integral of the lagrange polynomials for obejctive at 1.0
        if ncolloc == 1:
            self.lagpolyobjint1 = [1.0]
        else:
            self.lagpolyobjint_fun = lambda k, tau: np.sum(
                [self.lagpolyobj_coeff[k, p] * pow(tau, ncolloc - p) / (ncolloc - p) for p in range(ncolloc)]
            )
            self.lagpolyobjint1 = [self.lagpolyobjint_fun(k, 1.0) for k in range(ncolloc)]

    """
        Provides the coefficients for the lagrange polynomial as an array
        each row corresponds to coefficients of the lagrange polynomial at root k in {0,...,ncolloc}
        polynomial l_k(tau) = coeff[k,0]*x^{ncolloc+1} + coeff[k,1]*x^{ncolloc} + ... + coeff[k,ncolloc-1]*x + coeff[k,ncolloc]
    """

    def lagpoly_coeff(self, ncolloc, rx):
        coeff = np.ndarray((ncolloc + 1, ncolloc + 1))
        for k in range(ncolloc + 1):
            rx_k = [rx[j] for j in range(ncolloc + 1) if j != k]
            den = np.prod([(rx[k] - rxj) for rxj in rx_k])
            for p in range(ncolloc + 1):
                if p == 0:
                    coeff[k][p] = 1.0 / den
                elif p == 1:
                    coeff[k][p] = -np.sum(rx_k) / den
                elif p == ncolloc:
                    coeff[k][p] = np.prod(rx_k) * pow(-1.0, p) / den
                else:
                    combs = list(itt.combinations(rx_k, p))
                    coeff[k][p] = np.sum([np.prod(combs[i]) for i in range(len(combs))]) * pow(-1.0, p) / den
        return coeff


"""
        for k in range(ncolloc+1):
            rx_k = [rx[j] for j in range(ncolloc+1) if j != k]
            den = np.prod([(rx[k]-rxj) for rxj in rx_k])
            for p in range(ncolloc+1):
                if p == 0:
                    self.lagpolyx_coeff[k,p] = 1.0/den
                elif p == 1:
                    self.lagpolyx_coeff[k,p] = -np.sum(rx_k)/den
                elif p == ncolloc:
                    self.lagpolyx_coeff[k,p] = np.prod(rx_k)*pow(-1.0,p)/den
                else:
                    combs = list(itt.combinations(rx_k,p))
                    self.lagpolyx_coeff[k,p] = np.sum([np.prod(combs[i]) for i in range(len(combs))])*pow(-1.0,p)/den
#                    self.lagpolyx_coeff[k,p] = np.sum([np.prod([combs[i][j] for j in range(len(combs[i]))]) for i in range(len(combs))])*pow(-1.0,p)/den
"""
