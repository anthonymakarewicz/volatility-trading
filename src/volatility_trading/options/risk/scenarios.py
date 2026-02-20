"""Scenario generators used by option risk estimators.

Generators define shocked market states by emitting `StressScenario` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Protocol, runtime_checkable

from volatility_trading.options.risk.types import StressScenario
from volatility_trading.options.types import MarketShock, MarketState, OptionSpec


@runtime_checkable
class ScenarioGenerator(Protocol):
    """Contract for market-shock generation used by risk estimators."""

    def generate(
        self, *, spec: OptionSpec, state: MarketState
    ) -> tuple[StressScenario, ...]:
        """Return stress scenarios to evaluate for one option setup."""
        ...


@dataclass(frozen=True)
class FixedGridScenarioGenerator:
    """Generate a Cartesian grid of spot/vol/rate/time shocks.

    Notes:
        - `spot_shocks_pct` values are converted to absolute spot shocks via
          `state.spot * spot_shock_pct`.
        - scenario count is the Cartesian product size:
          `len(spot) * len(vol) * len(rate) * len(time)`.
    """

    spot_shocks_pct: tuple[float, ...] = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    vol_shocks: tuple[float, ...] = (-0.05, 0.0, 0.05)
    rate_shocks: tuple[float, ...] = (0.0,)
    time_shocks_years: tuple[float, ...] = (0.0,)
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if not self.spot_shocks_pct:
            raise ValueError("spot_shocks_pct must not be empty")
        if not self.vol_shocks:
            raise ValueError("vol_shocks must not be empty")
        if not self.rate_shocks:
            raise ValueError("rate_shocks must not be empty")
        if not self.time_shocks_years:
            raise ValueError("time_shocks_years must not be empty")

    def generate(
        self, *, spec: OptionSpec, state: MarketState
    ) -> tuple[StressScenario, ...]:
        """Build stress scenarios from the configured fixed shock grid."""
        # `spec` is intentionally part of the interface for generators that may
        # depend on moneyness or time-to-expiry; fixed-grid does not need it.
        _ = spec
        scenarios = []
        for d_spot_pct, d_vol, d_rate, dt_years in product(
            self.spot_shocks_pct,
            self.vol_shocks,
            self.rate_shocks,
            self.time_shocks_years,
        ):
            d_spot = state.spot * d_spot_pct
            name = (
                f"ds={d_spot_pct:+.1%}|dv={d_vol:+.4f}|dr={d_rate:+.4f}"
                f"|dt={dt_years:.6f}"
            )
            scenarios.append(
                StressScenario(
                    name=name,
                    shock=MarketShock(
                        d_spot=d_spot,
                        d_volatility=d_vol,
                        d_rate=d_rate,
                        dt_years=dt_years,
                    ),
                )
            )

        if not self.deduplicate:
            return tuple(scenarios)

        unique: dict[tuple[float, float, float, float], StressScenario] = {}
        for scenario in scenarios:
            key = (
                scenario.shock.d_spot,
                scenario.shock.d_volatility,
                scenario.shock.d_rate,
                scenario.shock.dt_years,
            )
            unique[key] = scenario
        return tuple(unique.values())
