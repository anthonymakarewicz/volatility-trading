import pytest

from volatility_trading.options import (
    BinomialTreePricer,
    GreeksModel,
    MarketState,
    OptionSpec,
    OptionType,
    PriceModel,
    bs_price,
)


def test_binomial_tree_pricer_is_price_model_only():
    pricer = BinomialTreePricer(steps=200, american=True)

    assert isinstance(pricer, PriceModel)
    assert not isinstance(pricer, GreeksModel)


@pytest.mark.parametrize(
    "option_type",
    [OptionType.CALL, OptionType.PUT],
)
def test_binomial_european_converges_to_black_scholes(option_type: OptionType):
    spec = OptionSpec(strike=100.0, time_to_expiry=45 / 365.0, option_type=option_type)
    state = MarketState(spot=102.0, volatility=0.25, rate=0.03, dividend_yield=0.01)

    tree = BinomialTreePricer(steps=800, american=False).price(spec, state)
    bs = bs_price(
        S=state.spot,
        K=spec.strike,
        T=spec.time_to_expiry,
        sigma=state.volatility,
        r=state.rate,
        q=state.dividend_yield,
        option_type=spec.option_type,
    )
    assert tree == pytest.approx(bs, abs=1e-2)


def test_american_put_is_not_below_european_put():
    spec = OptionSpec(
        strike=100.0,
        time_to_expiry=90 / 365.0,
        option_type=OptionType.PUT,
    )
    state = MarketState(spot=98.0, volatility=0.22, rate=0.02, dividend_yield=0.00)

    european = BinomialTreePricer(steps=600, american=False).price(spec, state)
    american = BinomialTreePricer(steps=600, american=True).price(spec, state)

    assert american >= european


def test_american_call_without_dividend_is_close_to_european():
    spec = OptionSpec(
        strike=100.0,
        time_to_expiry=120 / 365.0,
        option_type=OptionType.CALL,
    )
    state = MarketState(spot=103.0, volatility=0.2, rate=0.03, dividend_yield=0.0)

    european = BinomialTreePricer(steps=800, american=False).price(spec, state)
    american = BinomialTreePricer(steps=800, american=True).price(spec, state)

    assert american == pytest.approx(european, abs=5e-3)


def test_invalid_steps_raise():
    with pytest.raises(ValueError, match="steps must be >= 1"):
        BinomialTreePricer(steps=0)
