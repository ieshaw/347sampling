# -*- coding: utf-8 -*-


import numpy as np
import numpy.random as rd
from scipy.stats import norm
import time
from datetime import timedelta
import os
import multiprocessing
import itertools

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

def run_exp(T,mu):

    # Input
    n_MC = 10000
    n = 100
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



    t_start = time.time()

    print('Now on: mu = {}, T= {}'.format(mu, T))

    # Related coefficients
    theta = np.ceil(mu * n) / T
    CI_coef = norm.ppf(0.5 + CI_perc / 200.0)


    '''
    Importance sampling
    '''

    n_AR = 500000
    N_arr_AR = np.zeros(n_AR)

    default_counts = np.zeros(n)

    for i_AR in range(n_AR):
        # Step 1: initialize count and time
        t_sample = np.zeros(1)
        M = np.zeros(n)

        # Step 2: Calculate rate of transition and calculate a bounding rate
        p_wb = p_nocorr(t_sample, kappa, X0, c, gamma)
        p_bound = np.sum(p_wb)

        while t_sample[0] < T:
            # Step 3: Draw a random variable and accept/reject
            eps = rd.exponential(1 / p_bound, size=1)
            # For the CIR model with parameters given in the paper, p_wob is decreasing w.r.t time
            p_candidate_wob = np.multiply(p_nocorr(t_sample + eps, kappa, X0, c, gamma), 1 - M)
            p_candidate_wb = p_candidate_wob + np.matmul(beta, M)
            if p_bound * rd.uniform() <= np.sum(p_candidate_wb):
                # Step 4 (If there is a default): modify M
                conditional_prob = np.cumsum(p_candidate_wb) / np.sum(p_candidate_wb)
                ind = np.searchsorted(np.cumsum(p_candidate_wb) / np.sum(p_candidate_wb), rd.random())
                M[ind] = 1

            t_sample[0] = t_sample[0] + eps[0]

        default_counts[int(np.sum(M))] += 1

    Y_total = default_counts / np.sum(default_counts)

    output_dir = os.path.abspath(os.path.dirname(__file__)) + '/MC_csvs_100'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    p = 1 - np.sum(Y_total[:int(mu*n)])

    filename = output_dir + "/{}_{}.csv".format(mu, T)
    with open(filename, 'w') as f:
        #n_MC,n,T,,mu,mu_n,IS_Estimate,IS_Variance,95_CI,Time_spent
        f.write('n_MC,n,T,mu,mu_n,Estimate,Variance')
        f.write('\n{:.0f},{:.0f},{:.1f},{:.2f},{:.0f},{:e},{:e}'.format(
            n_MC,n,T,mu, mu*n, p, p * (1-p)))

    t_end = time.time()
    t_used = t_end - t_start
    print("Time spent = {0}".format(timedelta(seconds = t_used)))

T_list = np.array([0.5 ,1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

mu_list = (np.arange(20) + 1) * 10e-3

nprocs = multiprocessing.cpu_count()

grid = itertools.product(T_list, mu_list)

args = [iter(grid)] * nprocs

grid_zip = itertools.izip_longest(*args, fillvalue = None)

for test in grid_zip:
    procs = []
    out_q = multiprocessing.Queue()

    for pair in test:
        try:
            p = multiprocessing.Process(
                target=run_exp,
                args=(pair[0], pair[1]))
            procs.append(p)
            p.start()
        except:
            pass

