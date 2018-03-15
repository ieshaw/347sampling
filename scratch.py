

m = m_MC

#########
#Anything above is parallelized
#####


# Step 1 - Generate S

m = m_MC[i_MC]

S = np.sort(rd.uniform(0.0, T, m))
p_wob = p_nocorr(S, kappa, X0, c, gamma)

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


Y_total = Y_total * np.where(m_MC == 0, 0, 1)
