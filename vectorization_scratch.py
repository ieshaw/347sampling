# -*- coding: utf-8 -*-


import numpy as np
import numpy.random as rd
from scipy.stats import norm
import time
from datetime import timedelta
import os

rd.seed(seed=0)

'''
Functions
'''


# Get the left part of p without correlation terms beta given a time grid
#   - dimension len(t) * n
#   - for a row k in {1, ..., len(t)}, p_wob is the left side (without beta) of
#     [p_n^i(S_k, M_{S_k - 1}) for i = {1, ..., n}]
def p_nocorr(t, kappa, X0, c, gamma):
    exp_gamma_t = np.exp(np.outer(t, gamma))
    p_wob = (4 * X0 * np.square(gamma) * exp_gamma_t) / \
            np.square(gamma - kappa + (gamma + kappa) * exp_gamma_t) + \
            (2 * kappa * c * (exp_gamma_t - 1)) / \
            (gamma - kappa + (gamma + kappa) * exp_gamma_t)
    return p_wob

'''
Parameters
'''

# Input
n_MC = 10000
n = 1000
n_t = 100
T_range = np.array([1, 3, 5])
mu_range = 0.01 * np.arange(1, 21)
beta_scale_factor = 10
CI_perc = 95

# Model parameters
kappa = rd.uniform(0.5, 1.5, n)
X0 = c = rd.uniform(0.001, 0.051, n)
sigma_bar = rd.uniform(0, 0.2, n)
sigma = np.minimum(np.sqrt(2 * kappa * c), sigma_bar)
gamma = np.sqrt(np.square(kappa) + 2 * np.square(sigma))
beta = rd.uniform(0, 0.01, (n, n)) / beta_scale_factor
np.fill_diagonal(beta, 0)

'''
Usage
'''

T = 5
mu = 0.01

t_start = time.time()

print('Now on: mu = {}, T= {}'.format(mu, T))

# Related coefficients
theta = np.ceil(mu * n) / T
CI_coef = norm.ppf(0.5 + CI_perc / 200.0)


'''
Importance sampling
'''

# Store variables
Y_total = np.zeros(n_MC)
m_MC = rd.poisson(np.ceil(mu * n), n_MC)
#t_start = time.time()

print('here. note n_Mc = {}'.format(n_MC))


####
#Code I am working on
####

# Do Monte Carlo IS
for i_MC in range(n_MC):

    # Step 1 - Generate S

    m = m_MC[i_MC]
    # Go to the next iteration if no default
    if m == 0:
        continue
    S = np.sort(rd.uniform(0.0, T, m))
    p_wob = p_nocorr(S, kappa, X0, c, gamma)

    print('here i = {}'.format(i_MC))

    # Step 2 - Draw I and update M
    #   - p same as p_wob with the beta term which depends on M
    #   - Q same dimensions as p where the row k is
    #     [q_n^i(S_k, M_{S_k - 1}) / q_n(S_k, M_{S_k - 1}) for i = {1, ..., n}]
    #   - I[i] has the index at the which the transition (i+1) takes place in M

    p = np.zeros((m, n))
    Q = np.zeros((m, n))
    I = np.empty(0, dtype=int)
    for i in range(m):
        # Add correlation term in p
        p[i, :] = p_wob[i, :] + np.sum(beta[:, I], axis=1)
        # Absorbant states
        p[i, I] = 0
        # Compute Q
        Q[i, :] = p[i, :] / np.sum(p[i, :]) * theta
        Q[i, :] /= np.sum(Q[i, :])
        # Draw I
        I = np.append(I, np.searchsorted(np.cumsum(Q[i, :]), rd.random()))

    # Step 3 - Generate Y

    # Jump integral of D
    D1 = np.sum(np.log(T * np.sum(p, 1)))
    # Lebesgue integral of D
    D2 = 0
    t_grid = np.append(np.insert(S, 0, 0), T)
    t_grid_diff = np.diff(t_grid)
    for i in range(m + 1):
        t2 = np.arange(t_grid[i], t_grid[i + 1], t_grid_diff[i] / n_t)
        p2 = np.sum(p_nocorr(t2, kappa, X0, c, gamma)) + \
             n_t * np.sum(beta[:, I[:i]])
        D2 += p2 * t_grid_diff[i] / n_t

    # Generate Y
    minTSn = T if m < n else np.min(T, S[-1])
    Z = np.exp(minTSn * theta - m * np.log(np.ceil(mu * n)) + D1 - D2)
    Y = Z * (m >= mu * n)
    Y_total[i_MC] = Y

print('here out')

####
#End of Code I am working on
####

t_end = time.time()
t_used = t_end - t_start
print("Time spent = {0}".format(timedelta(seconds = t_used)))

filename = "Scratch_{0}_{1:.2f}.txt".format(T, mu)
with open(filename, 'w') as f:
    f.write("n_MC = {0:.0f}\n".format(n_MC))
    f.write("n = {0:.0f}\n".format(n))
    f.write("T = {0:.0f}\n".format(T))
    f.write("mu = {0:.2f}\n".format(mu))
    f.write("mu*n = {0:.0f}\n".format(mu*n))
    f.write("IS Estimate = {0:e}\n".format(np.mean(Y_total)))
    f.write("IS Variance = {0:e}\n".format(np.var(Y_total)))
    f.write("{0}% CI = {1:e}\n".format(CI_perc,
                                       CI_coef*np.sqrt(np.var(Y_total)/n_MC)))
    f.write("Time spent = {0}\n".format(timedelta(seconds = t_used)))



