import pytest

from volatility_trading.options import (
    BlackScholesPricer,
    GreekApproxPricer,
    GreeksModel,
    MarketShock,
    MarketState,
    OptionSpec,
    OptionType,
    OptionTypeInput,
    PriceModel,
    bs_greeks,
    bs_price,
)


def test_black_scholes_pricer_matches_functional_greeks_api():
    spec = OptionSpec(
        strike=100.0,
        time_to_expiry=30 / 365.0,
        option_type=OptionType.CALL,
    )
    state = MarketState(spot=101.0, volatility=0.22, rate=0.03, dividend_yield=0.015)

    pricer = BlackScholesPricer()
    out = pricer.price_and_greeks(spec, state)
    ref = bs_greeks(
        S=state.spot,
        K=spec.strike,
        T=spec.time_to_expiry,
        sigma=state.volatility,
        r=state.rate,
        q=state.dividend_yield,
        option_type=spec.option_type,
    )

    assert out.price == pytest.approx(ref["price"])
    assert out.greeks.delta == pytest.approx(ref["delta"])
    assert out.greeks.gamma == pytest.approx(ref["gamma"])
    assert out.greeks.vega == pytest.approx(ref["vega"])
    assert out.greeks.theta == pytest.approx(ref["theta"])
    assert out.delta == pytest.approx(ref["delta"])
    assert out.gamma == pytest.approx(ref["gamma"])
    assert out.vega == pytest.approx(ref["vega"])
    assert out.theta == pytest.approx(ref["theta"])
    assert out.rho == pytest.approx(ref["rho"])


def test_protocol_split_price_model_vs_greeks_model():
    class PriceOnlyPricer:
        def price(self, spec: OptionSpec, state: MarketState) -> float:
            return 0.0

    bs = BlackScholesPricer()
    price_only = PriceOnlyPricer()

    assert isinstance(bs, PriceModel)
    assert isinstance(bs, GreeksModel)
    assert isinstance(price_only, PriceModel)
    assert not isinstance(price_only, GreeksModel)


def test_greek_approx_pricer_close_to_exact_for_small_shocks():
    spec = OptionSpec(
        strike=100.0,
        time_to_expiry=30 / 365.0,
        option_type=OptionType.PUT,
    )
    state = MarketState(spot=100.0, volatility=0.20, rate=0.02, dividend_yield=0.01)
    shock = MarketShock(d_spot=-0.5, d_volatility=0.002, d_rate=0.0, dt_years=1 / 365.0)

    approx = GreekApproxPricer()
    approx_price = approx.price_with_shock(spec, state, shock)

    exact_price = bs_price(
        S=state.spot + shock.d_spot,
        K=spec.strike,
        T=max(spec.time_to_expiry - shock.dt_years, 1e-12),
        sigma=state.volatility + shock.d_volatility,
        r=state.rate + shock.d_rate,
        q=state.dividend_yield,
        option_type=spec.option_type,
    )

    assert approx_price == pytest.approx(exact_price, abs=2e-3)


@pytest.mark.parametrize(
    ("option_type", "expected_positive"),
    [("C", True), ("P", True), ("call", True), ("put", True)],
)
def test_legacy_bs_price_supports_upper_and_lower_option_types(
    option_type: OptionTypeInput, expected_positive: bool
):
    price = bs_price(S=100.0, K=100.0, T=30 / 365.0, sigma=0.2, option_type=option_type)
    assert (price > 0) is expected_positive
