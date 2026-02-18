from __future__ import annotations

from collections.abc import Sequence

# TODO: add term-structure helpers here once we refactor them out of the skew trading notebook:
#   * pick_quote_row(...)
#   * interp_iv(...)
#   * find_viable_dtes(...)


def pick_closest_dte(
    dtes: Sequence[int],
    target: int,
    max_tol: int = 5,
) -> int | None:
    """Pick DTE closest to target within max_tol, else None."""
    if not dtes:
        return None
    best = min(dtes, key=lambda d: abs(d - target))
    return best if abs(best - target) <= max_tol else None
