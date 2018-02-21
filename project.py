# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import time

rd.seed(seed = 0)


'''
Parameters
'''


# Input
n_MC = 10000
n = 100
n_t = 100
T_range = np.array([1, 3, 5])
mu_range = 0.01 * np.arange(1, 21)
beta_scale_factor = 10

# Model parameters
kappa = rd.uniform(0.5, 1.5, n)
X0 = c = rd.uniform(0.001, 0.051, n)
sigma_bar = rd.uniform(0, 0.2, n)
sigma = np.minimum(np.sqrt(2 * kappa * c), sigma_bar)
gamma = np.sqrt(np.square(kappa) + 2 * np.square(sigma))
beta = rd.uniform(0, 0.01, (n, n)) / beta_scale_factor
np.fill_diagonal(beta, 0)

# Choose input
T = T_range[0]
mu = mu_range[-1]
theta = np.ceil(mu*n) / T


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
Importance sampling
'''


# Store variables
Y_total = np.zeros(n_MC)
t_start = time.time()

# Do Monte Carlo IS
for i_MC in range(n_MC):

    # Step 1 - Generate S
    # TODO - change S generation scheme
    
    S = np.cumsum(rd.exponential(scale = 1/theta,      # beta = 1/lambda
                                 size = int(5*theta))) # E[N] = lambda, x5 to make sure
    S = S[S <= T][:n]
    m = int(S.shape[0])
    p_wob = p_nocorr(S, kappa, X0, c, gamma)
    
    # Step 2 - Draw I and update M
    #   - p same as p_wob with the beta term which depends on M
    #   - Q same dimensions as p where the row k is
    #     [q_n^i(S_k, M_{S_k - 1}) / q_n(S_k, M_{S_k - 1}) for i = {1, ..., n}]
    #   - I[i] has the index at the which the transition (i+1) takes place in M
    
    p = np.zeros((m, n))
    Q = np.zeros((m, n))
    I = np.empty(0, dtype = int)
    for i in range(m):
        # Add correlation term in p
        p[i, :] = p_wob[i, :] + np.sum(beta[:, I], axis = 1)
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
    for i in range(m+1):
        t2 = np.arange(t_grid[i], t_grid[i+1], t_grid_diff[i] / n_t)
        p2 = np.sum(p_nocorr(t2, kappa, X0, c, gamma)) + \
             n_t * np.sum(beta[:, I[:i]])
        D2 += p2 * t_grid_diff[i] / n_t
        
    # Generate Y
    minTSn = T if m < n else np.min(T, S[-1])
    Z = np.exp(minTSn*theta - m*np.log(np.ceil(mu*n)) + D1 - D2)
    Y = Z * (m >= mu*n)
    Y_total[i_MC] = Y
    
    # Print progress
    if (i_MC+1) % (n_MC/20) == 0:
        print "Iteration {0}/{1} in {2}s".format(i_MC+1, n_MC, 
                                                 time.time() - t_start)
    

'''
Results
'''

print "mu*n = {0}".format(mu*n) 
print "IS Estimate = {0}".format(np.mean(Y_total))
print "IS Variance = {0}".format(np.var(Y_total))
    
