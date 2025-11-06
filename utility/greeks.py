import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_delta_call(S, K, T, sigma, r=0.0, q=0.0):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


def bs_delta_put(S, K, T, sigma, r=0.0, q=0.0):
    # put delta = call delta - e^{-qT}
    return bs_delta_call(S, K, T, sigma, r, q) - np.exp(-q * T)


def solve_strike_for_delta(target_delta, S, T, sigma, option_type, r=0.0, q=0.0, 
                            K_min_factor=0.2, K_max_factor=3.0, tol=1e-6, max_iter=100):
    """
    Solve for strike K such that the (spot) delta equals target_delta.
    Uses bisection with a fixed sigma (e.g. ATM vol) as approximation.
    """
    K_low = K_min_factor * S
    K_high = K_max_factor * S

    if option_type == "call":
        delta_low = bs_delta_call(S, K_low, T, sigma, r, q)
        delta_high = bs_delta_call(S, K_high, T, sigma, r, q)
    else:  # put
        delta_low = bs_delta_put(S, K_low, T, sigma, r, q)
        delta_high = bs_delta_put(S, K_high, T, sigma, r, q)

    # Ensure target_delta is within bracket; if not, just clip
    if not (min(delta_low, delta_high) <= target_delta <= max(delta_low, delta_high)):
        # simple fallback: return ATM
        return S

    for _ in range(max_iter):
        K_mid = 0.5 * (K_low + K_high)
        if option_type == "call":
            delta_mid = bs_delta_call(S, K_mid, T, sigma, r, q)
        else:
            delta_mid = bs_delta_put(S, K_mid, T, sigma, r, q)

        if abs(delta_mid - target_delta) < tol:
            return K_mid

        # Bisection step
        if (delta_low - target_delta) * (delta_mid - target_delta) <= 0:
            K_high = K_mid
            delta_high = delta_mid
        else:
            K_low = K_mid
            delta_low = delta_mid

    return K_mid  # last iterate if not converged
