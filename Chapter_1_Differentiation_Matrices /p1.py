#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2021/3/6 19:48
"""
# p1.py - convergence of fourth-order finite differences
For various n, set up grid in [-pi, pi] and function u(x)=e^{sin(x)}
"""

import numpy as np
from numpy import pi
from numpy import sin
from numpy import cos
from numpy import exp
from scipy.linalg import norm
from scipy.sparse import dia_matrix
from matplotlib import pyplot as plt


def fourth_order(n):
    h = 2 * pi / n
    x = -pi + np.arange(1, n+1)*h
    u = exp(sin(x))
    u_prime = cos(x) * u
    e = np.ones(n, dtype=np.float)
    data = np.array([2*e/3, -e/12, e/12, -2*e/3])
    offsets = np.array([1, 2, n-2, n-1])
    d = dia_matrix((data, offsets), shape=(n, n))
    derivative = (d - d.transpose())/h
    w = derivative * u
    e = norm(w-u_prime, ord=np.inf)
    return e


if __name__ == "__main__":
    n_vec = 2 ** np.arange(3, 13)
    length = len(n_vec)
    error = np.zeros(length)
    for i in range(length):
        error[i] = fourth_order(n_vec[i])
    n_vec_f = np.array([np.float(n) for n in n_vec])
    plt.figure()
    plt.loglog(n_vec, error, '*-')
    plt.loglog(n_vec, n_vec_f ** (-4))
    plt.title("Convergence of 4th-order finite difference")
    plt.show()
