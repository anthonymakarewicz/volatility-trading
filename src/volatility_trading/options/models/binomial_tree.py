"""CRR binomial-tree pricing for vanilla options."""

from __future__ import annotations

import numpy as np

from volatility_trading.options.models.black_scholes import normalize_option_type
from volatility_trading.options.types import OptionType, OptionTypeInput

# TODO(dividends): Replace flat continuous-yield treatment with
# discrete cash dividends. Prefer `nextDiv` from the ORATS core endpoint;
# infer ex-div date from ORATS fields when possible, otherwise fall back to
# external ex-div calendars (e.g., yfinance) for schedule alignment.


def _intrinsic_value(
    spot: np.ndarray | float, strike: float, option_type: OptionType
) -> np.ndarray:
    if option_type == OptionType.CALL:
        return np.maximum(np.asarray(spot, dtype=float) - strike, 0.0)
    return np.maximum(strike - np.asarray(spot, dtype=float), 0.0)


def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: OptionTypeInput = OptionType.CALL,
    steps: int = 200,
    american: bool = True,
) -> float:
    """Price a vanilla option with a Cox-Ross-Rubinstein tree.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        sigma: Annualized volatility in decimals.
        r: Continuously-compounded risk-free rate.
        q: Continuously-compounded dividend yield (current approximation).
        option_type: One of `{'call', 'put', 'C', 'P'}`.
        steps: Number of binomial time steps.
        american: If True, allow early exercise at each node.

    Returns:
        Present value for one option.

    Raises:
        ValueError: If `steps < 1`, if `T < 0`, or if CRR probabilities become
            invalid for the selected parameters.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if T < 0:
        raise ValueError("T must be non-negative")

    opt_type = normalize_option_type(option_type)
    if T == 0:
        return float(_intrinsic_value(S, K, opt_type))

    dt = T / steps
    disc = np.exp(-r * dt)

    # Degenerate path: deterministic evolution when volatility is zero.
    if sigma <= 0:
        times = np.arange(steps + 1) * dt
        spots = S * np.exp((r - q) * times)
        intrinsic = _intrinsic_value(spots, K, opt_type)

        if american:
            discounted = np.exp(-r * times) * intrinsic
            return float(discounted.max())

        terminal_payoff = float(intrinsic[-1])
        return float(np.exp(-r * T) * terminal_payoff)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "Invalid CRR risk-neutral probability; increase steps or check inputs."
        )

    j = np.arange(steps + 1)
    spots_T = S * (u**j) * (d ** (steps - j))
    option_vals = _intrinsic_value(spots_T, K, opt_type)

    for step in range(steps - 1, -1, -1):
        option_vals = disc * (p * option_vals[1:] + (1.0 - p) * option_vals[:-1])
        if not american:
            continue

        j = np.arange(step + 1)
        spots = S * (u**j) * (d ** (step - j))
        intrinsic = _intrinsic_value(spots, K, opt_type)
        option_vals = np.maximum(option_vals, intrinsic)

    return float(option_vals[0])
