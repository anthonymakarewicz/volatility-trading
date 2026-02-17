"""Black-Scholes pricing and Greeks for European options."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from volatility_trading.options.types import CanonicalOptionType, OptionType


def normalize_option_type(option_type: OptionType) -> CanonicalOptionType:
    """Normalize option type labels to 'call'/'put'."""
    if option_type in ("call", "put"):
        return option_type
    if option_type == "C":
        return "call"
    if option_type == "P":
        return "put"
    raise ValueError("option_type must be one of {'call', 'put', 'C', 'P'}")


def bs_d1_d2(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes with continuous dividend yield."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes price with continuous dividend yield."""
    opt_type = normalize_option_type(option_type)
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)

    if opt_type == "call":
        return float(
            S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        )
    return float(
        K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    )


def bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes delta."""
    opt_type = normalize_option_type(option_type)
    if sigma <= 0 or T <= 0:
        return 0.0

    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    delta_call = np.exp(-q * T) * norm.cdf(d1)
    if opt_type == "call":
        return float(delta_call)
    return float(delta_call - np.exp(-q * T))


def bs_gamma(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Black-Scholes gamma."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Black-Scholes vega per +1.0 volatility."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def bs_theta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes theta per +1.0 calendar year."""
    opt_type = normalize_option_type(option_type)
    if sigma <= 0 or T <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if opt_type == "call":
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return float(term1 + term2 + term3)


def bs_rho(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes rho per +1.0 rate."""
    opt_type = normalize_option_type(option_type)
    if sigma <= 0 or T <= 0:
        return 0.0
    _, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    if opt_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2))
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> dict[str, float]:
    """Return Black-Scholes price and Greeks for one option."""
    return {
        "price": bs_price(S, K, T, sigma, r, q, option_type),
        "delta": bs_delta(S, K, T, sigma, r, q, option_type),
        "gamma": bs_gamma(S, K, T, sigma, r, q),
        "vega": bs_vega(S, K, T, sigma, r, q),
        "theta": bs_theta(S, K, T, sigma, r, q, option_type),
        "rho": bs_rho(S, K, T, sigma, r, q, option_type),
    }


def solve_strike_for_delta(
    target_delta: float,
    S: float,
    T: float,
    sigma: float,
    option_type: OptionType,
    r: float = 0.0,
    q: float = 0.0,
    K_min_factor: float = 0.2,
    K_max_factor: float = 3.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Solve strike K such that BS spot delta equals `target_delta`."""
    opt_type = normalize_option_type(option_type)
    K_low = K_min_factor * S
    K_high = K_max_factor * S

    delta_low = bs_delta(S, K_low, T, sigma, r, q, option_type=opt_type)
    delta_high = bs_delta(S, K_high, T, sigma, r, q, option_type=opt_type)

    # Ensure target delta is bracketed; fallback to ATM otherwise.
    if not (min(delta_low, delta_high) <= target_delta <= max(delta_low, delta_high)):
        return S

    K_mid = S
    for _ in range(max_iter):
        K_mid = 0.5 * (K_low + K_high)
        delta_mid = bs_delta(S, K_mid, T, sigma, r, q, option_type=opt_type)

        if abs(delta_mid - target_delta) < tol:
            return float(K_mid)

        if (delta_low - target_delta) * (delta_mid - target_delta) <= 0:
            K_high = K_mid
            delta_high = delta_mid
        else:
            K_low = K_mid
            delta_low = delta_mid

    return float(K_mid)
