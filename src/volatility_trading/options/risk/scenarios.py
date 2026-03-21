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
          `len(spot) * len(vol) * len(rr) * len(rate) * len(time)`.
    """

    spot_shocks_pct: tuple[float, ...] = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    vol_shocks: tuple[float, ...] = (-0.05, 0.0, 0.05)
    risk_reversal_shocks: tuple[float, ...] = (0.0,)
    rate_shocks: tuple[float, ...] = (0.0,)
    time_shocks_years: tuple[float, ...] = (0.0,)
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if not self.spot_shocks_pct:
            raise ValueError("spot_shocks_pct must not be empty")
        if not self.vol_shocks:
            raise ValueError("vol_shocks must not be empty")
        if not self.risk_reversal_shocks:
            raise ValueError("risk_reversal_shocks must not be empty")
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
        for d_spot_pct, d_vol, d_risk_reversal, d_rate, dt_years in product(
            self.spot_shocks_pct,
            self.vol_shocks,
            self.risk_reversal_shocks,
            self.rate_shocks,
            self.time_shocks_years,
        ):
            d_spot = state.spot * d_spot_pct
            name = (
                f"ds={d_spot_pct:+.1%}|dv={d_vol:+.4f}|dr={d_rate:+.4f}"
                f"|drr={d_risk_reversal:+.4f}|dt={dt_years:.6f}"
            )
            scenarios.append(
                StressScenario(
                    name=name,
                    shock=MarketShock(
                        d_spot=d_spot,
                        d_volatility=d_vol,
                        d_risk_reversal=d_risk_reversal,
                        d_rate=d_rate,
                        dt_years=dt_years,
                    ),
                )
            )

        if not self.deduplicate:
            return tuple(scenarios)

        unique: dict[tuple[float, float, float, float, float], StressScenario] = {}
        for scenario in scenarios:
            key = (
                scenario.shock.d_spot,
                scenario.shock.d_volatility,
                scenario.shock.d_risk_reversal,
                scenario.shock.d_rate,
                scenario.shock.dt_years,
            )
            unique[key] = scenario
        return tuple(unique.values())


@dataclass(frozen=True)
class _NamedScenarioDefinition:
    """One reusable named stress scenario template."""

    name: str
    d_spot_pct: float = 0.0
    d_volatility: float = 0.0
    d_risk_reversal: float = 0.0
    d_rate: float = 0.0
    dt_years: float = 0.0


_NAMED_SCENARIO_FAMILIES: dict[str, tuple[_NamedScenarioDefinition, ...]] = {
    "core": (
        _NamedScenarioDefinition(
            name="core.selloff_mild",
            d_spot_pct=-0.05,
            d_volatility=0.02,
        ),
        _NamedScenarioDefinition(
            name="core.selloff_severe",
            d_spot_pct=-0.10,
            d_volatility=0.05,
        ),
        _NamedScenarioDefinition(
            name="core.rally_mild",
            d_spot_pct=0.05,
            d_volatility=-0.02,
        ),
        _NamedScenarioDefinition(
            name="core.rally_strong",
            d_spot_pct=0.10,
            d_volatility=-0.04,
        ),
        _NamedScenarioDefinition(name="core.vol_up", d_volatility=0.05),
        _NamedScenarioDefinition(name="core.vol_down", d_volatility=-0.05),
    ),
    "tail": (
        _NamedScenarioDefinition(
            name="tail.crash_extreme",
            d_spot_pct=-0.15,
            d_volatility=0.08,
        ),
        _NamedScenarioDefinition(
            name="tail.rally_extreme",
            d_spot_pct=0.15,
            d_volatility=-0.05,
        ),
    ),
    "rr": (
        _NamedScenarioDefinition(
            name="rr.steepen_mild",
            d_risk_reversal=-0.03,
        ),
        _NamedScenarioDefinition(
            name="rr.steepen_severe",
            d_risk_reversal=-0.05,
        ),
        _NamedScenarioDefinition(
            name="rr.flatten_mild",
            d_risk_reversal=0.03,
        ),
        _NamedScenarioDefinition(
            name="rr.flatten_severe",
            d_risk_reversal=0.05,
        ),
        _NamedScenarioDefinition(
            name="rr.selloff_steepen",
            d_spot_pct=-0.10,
            d_volatility=0.05,
            d_risk_reversal=-0.05,
        ),
        _NamedScenarioDefinition(
            name="rr.crash_steepen_extreme",
            d_spot_pct=-0.15,
            d_volatility=0.08,
            d_risk_reversal=-0.05,
        ),
        _NamedScenarioDefinition(
            name="rr.rally_flatten",
            d_spot_pct=0.10,
            d_volatility=-0.03,
            d_risk_reversal=0.05,
        ),
        _NamedScenarioDefinition(
            name="rr.rally_flatten_extreme",
            d_spot_pct=0.15,
            d_volatility=-0.05,
            d_risk_reversal=0.05,
        ),
    ),
}


def available_named_scenario_families() -> tuple[str, ...]:
    """Return the stable set of built-in named stress scenario families."""
    return tuple(sorted(_NAMED_SCENARIO_FAMILIES))


@dataclass(frozen=True)
class NamedScenarioGenerator:
    """Generate curated named economic scenarios from reusable families.

    The generator stays shared and strategy-agnostic: callers select one or
    more scenario families, and each family contributes a predefined set of
    economically interpretable stress scenarios.
    """

    scenario_families: tuple[str, ...] = ("core",)
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if not self.scenario_families:
            raise ValueError("scenario_families must not be empty")
        unknown = tuple(
            family
            for family in self.scenario_families
            if family not in _NAMED_SCENARIO_FAMILIES
        )
        if unknown:
            available = ", ".join(available_named_scenario_families())
            unknown_text = ", ".join(sorted(set(unknown)))
            raise ValueError(
                "Unknown scenario_families: "
                f"{unknown_text}. Available families: {available}."
            )

    def generate(
        self, *, spec: OptionSpec, state: MarketState
    ) -> tuple[StressScenario, ...]:
        """Build named stress scenarios from the selected built-in families."""
        _ = spec
        scenarios = tuple(
            StressScenario(
                name=definition.name,
                shock=MarketShock(
                    d_spot=state.spot * definition.d_spot_pct,
                    d_volatility=definition.d_volatility,
                    d_risk_reversal=definition.d_risk_reversal,
                    d_rate=definition.d_rate,
                    dt_years=definition.dt_years,
                ),
            )
            for family in self.scenario_families
            for definition in _NAMED_SCENARIO_FAMILIES[family]
        )

        if not self.deduplicate:
            return scenarios

        unique: dict[tuple[float, float, float, float, float], StressScenario] = {}
        for scenario in scenarios:
            key = (
                scenario.shock.d_spot,
                scenario.shock.d_volatility,
                scenario.shock.d_risk_reversal,
                scenario.shock.d_rate,
                scenario.shock.dt_years,
            )
            unique.setdefault(key, scenario)
        return tuple(unique.values())
