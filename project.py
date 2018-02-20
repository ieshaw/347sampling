# -*- coding: utf-8 -*-


import numpy as np
import numpy.random as rd
import os

rd.seed(seed = 0)


'''
Parameters
'''


# Input
n_MC = 2
n = 1000
T_range = np.array([1, 3, 5])
mu_range = np.arange(0.01, 0.2 + 0.001, 0.01)
beta_scale_factor = 10

# Model parameters
kappa = rd.uniform(0.5, 1.5, n)
X0 = c = rd.uniform(0.001, 0.51, n)
sigma_bar = rd.uniform(0, 0.2, n)
sigma = np.minimum(np.sqrt(2 * kappa * c), sigma_bar)
gamma = np.sqrt(np.square(kappa) + 2 * np.square(sigma))
beta = rd.uniform(0, 0.01, (n, n)) / beta_scale_factor
np.fill_diagonal(beta, 0)


'''
Importance sampling
'''


# Choose input
T = T_range[0]
mu = mu_range[0]
theta = np.ceil(mu*n) / T

# Step 1 - Generate S
S = np.cumsum(rd.exponential(scale = 1/theta,      # beta = 1/lambda
                             size = int(5*theta))) # E[N] = lambda, x5 to make sure
S = S[S <= T][:n]
m = int(S.shape[0])

# p without the correlation factor beta
#   - p_wob is the left side of p
#       - dimension m * n
#       - for a row k in {1, ..., m}, p_wob is the left side (without beta) of 
#       [p_n^i(S_k, M_{S_k - 1}) for i = {1, ..., n}]
exp_gamma_t = np.exp(np.outer(S, gamma))
p_wob = (4 * X0 * np.square(gamma) * exp_gamma_t) / \
            np.square(gamma - kappa + (gamma + kappa) * exp_gamma_t) + \
        (2 * kappa * c * (exp_gamma_t - 1)) / \
            (gamma - kappa + (gamma + kappa) * exp_gamma_t)

# Step 2 - Draw I and update M
#   - p same as p_wob with the beta term which depends on M
#   - Q same dimensions as p where the row k is
#     [q_n^i(S_k, M_{S_k - 1}) / q_n(S_k, M_{S_k - 1}) for i = {1, ..., n}]
#   - M is our default markov chain
#       - dimension (m+1) * n
#       - row 0 corresponds to M_0 which is used in the for loop
#   - I has the index at the which the transition takes place in M
p = np.zeros((m, n))
Q = np.zeros((m, n))
M = np.zeros((m+1, n), dtype = int)
I = np.zeros(m, dtype = int)
for i in range(m):
    # Add correlation term in p
    p[i, :] = p_wob[i, :] + beta.dot(M[i, :])
    # Absorbant state
    p[i, M[i, :] != 0] = 0
    # Compute Q
    Q[i, :] = p[i, :] / np.sum(p[i, :]) * theta
    Q[i, :] /= np.sum(Q[i, :])
    # Draw I
    I[i] = np.searchsorted(np.cumsum(Q[i, :]), rd.random())
    # Update M
    M[(i+1):, I[i]] = 1

# Step 3 - Generate Y
# TODO - Formula for D
p_n = np.sum(p, 1)
#D = np.sum(p_n) - 0
D = 0
minTSn = T if m < n else np.min(T, S[-1])
Z = np.exp(minTSn*theta - I.shape[0] * np.log(np.ceil(mu*n)) + D)
Y = Z * (I.shape[0] >= mu*n)


