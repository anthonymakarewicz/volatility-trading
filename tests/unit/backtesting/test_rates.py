import pandas as pd
import pytest

from volatility_trading.backtesting.rates import (
    ConstantRateModel,
    SeriesRateModel,
    coerce_rate_model,
)


def test_constant_rate_model_returns_constant_value():
    model = ConstantRateModel(rate_annual=0.03)

    assert model.annual_rate() == pytest.approx(0.03)
    assert model.annual_rate(pd.Timestamp("2020-01-01")) == pytest.approx(0.03)


def test_series_rate_model_uses_asof_lookup_and_first_value_floor():
    series = pd.Series(
        [0.01, 0.015, 0.02],
        index=pd.to_datetime(["2020-01-02", "2020-01-10", "2020-01-20"]),
    )
    model = SeriesRateModel(series)

    assert model.annual_rate(pd.Timestamp("2019-12-31")) == pytest.approx(0.01)
    assert model.annual_rate(pd.Timestamp("2020-01-15")) == pytest.approx(0.015)
    assert model.annual_rate(pd.Timestamp("2020-01-25")) == pytest.approx(0.02)
    assert model.annual_rate() == pytest.approx(0.02)


def test_coerce_rate_model_accepts_numeric_and_series():
    numeric_model = coerce_rate_model(0.025)
    series_model = coerce_rate_model(
        pd.Series([0.01, 0.02], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    )

    assert numeric_model.annual_rate(pd.Timestamp("2020-01-01")) == pytest.approx(0.025)
    assert series_model.annual_rate(pd.Timestamp("2020-01-02")) == pytest.approx(0.02)
