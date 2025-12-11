import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_d1_d2(S, K, T, sigma, r=0.0, q=0.0):
    """Compute d1 and d2 for Black–Scholes with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, sigma, r=0.0, q=0.0, option_type='call'):
    """Black-Scholes formula with dividend yield"""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)

    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Not a valid option type, type must be 'call' or 'put'")


def bs_delta(S, K, T, sigma, r=0.0, q=0.0, option_type='call'):
    if sigma <= 0 or T <= 0:
        return 0.0

    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    delta_call = np.exp(-q * T) * norm.cdf(d1)

    if option_type == "call":
        return delta_call
    elif option_type == "put":
        return delta_call - np.exp(-q * T)  # = e^{-qT}(N(d1) - 1)
    else:
        raise ValueError("Not a valid option type, type must be 'call' or 'put'")


def bs_gamma(S, K, T, sigma, r=0.0, q=0.0):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, sigma, r=0.0, q=0.0):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)  # per 1.0 vol


def bs_theta(S, K, T, sigma, r=0.0, q=0.0, option_type='call'):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    term1 = - (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        term2 = - q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return term1 + term2 + term3 # per 1.0 YTE


def bs_rho(S, K, T, sigma, r=0.0, q=0.0, option_type='call'):
    if sigma <= 0 or T <= 0:
        return 0.0
    _, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    

def bs_greeks(S, K, T, sigma, r=0.0, q=0.0, option_type="call"):
    """
    Convenience wrapper: return price and all Greeks for a given option.

    Returns a dict with:
        price, delta, gamma, vega, theta, rho
    """
    return {
        "price": bs_price(S, K, T, sigma, r, q, option_type),
        "delta": bs_delta(S, K, T, sigma, r, q, option_type),
        "gamma": bs_gamma(S, K, T, sigma, r, q),
        "vega":  bs_vega(S, K, T, sigma, r, q),
        "theta": bs_theta(S, K, T, sigma, r, q, option_type),
        "rho":   bs_rho(S, K, T, sigma, r, q, option_type),
    }


def solve_strike_for_delta(
    target_delta, 
    S, 
    T, 
    sigma, 
    option_type, 
    r=0.0, 
    q=0.0, 
    K_min_factor=0.2, 
    K_max_factor=3.0, 
    tol=1e-6, 
    max_iter=100
):
    """
    Solve for strike K such that the (spot) delta equals target_delta.
    Uses bisection with a fixed sigma (e.g. ATM vol) as approximation.
    """
    K_low = K_min_factor * S
    K_high = K_max_factor * S

    if option_type == "call":
        delta_low = bs_delta(S, K_low, T, sigma, r, q, option_type="call")
        delta_high = bs_delta(S, K_high, T, sigma, r, q, option_type="call")
    else:  # put
        delta_low = bs_delta(S, K_low, T, sigma, r, q, option_type="put")
        delta_high = bs_delta(S, K_high, T, sigma, r, q, option_type="put")

    # Ensure target_delta is within bracket; if not, just clip
    if not (min(delta_low, delta_high) <= target_delta <= max(delta_low, delta_high)):
        # simple fallback: return ATM
        return S

    for _ in range(max_iter):
        K_mid = 0.5 * (K_low + K_high)
        if option_type == "call":
            delta_mid = bs_delta(S, K_mid, T, sigma, r, q, option_type="call")
        else:
            delta_mid = bs_delta(S, K_mid, T, sigma, r, q, option_type="put")

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