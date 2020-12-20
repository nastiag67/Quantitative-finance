import numpy as np
import math


def simulate_tree_np(M, T, S0, sigma):
    """
    BINOMIAL OPTION PRICING
    S0 - Initial value of the risky asset.
    T - Time horizon for the binomial tree simulation.
    r - Constant short rate.
    sigma - Constant volatility factor.
    """
    # Length of the time intervals
    dt = T / M
    # Factors for the upward movements (ndarray object with gross upward movements)
    up = np.arange(M + 1)
    up = np.resize(up, (M + 1, M + 1))
    # Factors for the downward movements (ndarray object with gross downward movements)
    down = up.transpose() * 2
    # Tree for four time intervals (upper-right triangle of values)
    S = S0 * np.exp(sigma * math.sqrt(dt) * (up - down))
    return S


def mcs_simulation_np(M, T, S0, I, sigma, r):
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    # The random numbers, drawn in a single vectorized step
    rn = np.random.standard_normal(S.shape)
    # The loop over the time intervals
    for t in range(1, M + 1):
        # The Euler scheme with vectorized NumPy code handling all paths at once.
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
    return S

