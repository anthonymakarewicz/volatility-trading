import pandas as pd
import pytest

from volatility_trading.backtesting import MarginAccount, MarginPolicy


def test_margin_account_no_call_when_equity_above_maintenance():
    account = MarginAccount(MarginPolicy(maintenance_margin_ratio=0.75))

    status = account.evaluate(
        equity=10_000.0,
        initial_margin_requirement=6_000.0,
        open_contracts=2,
    )

    assert status.core.maintenance_margin_requirement == pytest.approx(4_500.0)
    assert status.core.margin_excess == pytest.approx(5_500.0)
    assert status.core.in_margin_call is False
    assert status.core.margin_call_days == 0
    assert status.core.forced_liquidation is False


def test_margin_account_forces_full_liquidation_after_grace():
    account = MarginAccount(
        MarginPolicy(
            maintenance_margin_ratio=0.75,
            margin_call_grace_days=1,
            liquidation_mode="full",
        )
    )

    day1 = account.evaluate(
        equity=4_000.0,
        initial_margin_requirement=8_000.0,
        open_contracts=2,
    )
    day2 = account.evaluate(
        equity=4_000.0,
        initial_margin_requirement=8_000.0,
        open_contracts=2,
    )

    assert day1.core.in_margin_call is True
    assert day1.core.forced_liquidation is False
    assert day1.core.margin_call_days == 1
    assert day2.core.in_margin_call is True
    assert day2.core.forced_liquidation is True
    assert day2.contracts_to_liquidate == 2
    assert day2.contracts_after_liquidation == 0


def test_margin_account_target_liquidation_uses_buffer():
    account = MarginAccount(
        MarginPolicy(
            maintenance_margin_ratio=0.75,
            margin_call_grace_days=0,
            liquidation_mode="target",
            liquidation_buffer_ratio=0.10,
        )
    )

    status = account.evaluate(
        equity=7_000.0,
        initial_margin_requirement=12_000.0,
        open_contracts=4,
    )

    assert status.core.forced_liquidation is True
    assert status.contracts_after_liquidation == 2
    assert status.contracts_to_liquidate == 2


def test_margin_account_financing_terms_for_cash_and_borrow():
    policy = MarginPolicy(
        apply_financing=True,
        cash_rate_annual=0.02,
        borrow_rate_annual=0.06,
        trading_days_per_year=252,
    )
    account = MarginAccount(policy)

    borrow_status = account.evaluate(
        equity=8_000.0,
        initial_margin_requirement=10_000.0,
        open_contracts=1,
    )
    cash_status = account.evaluate(
        equity=12_000.0,
        initial_margin_requirement=10_000.0,
        open_contracts=1,
    )

    assert borrow_status.borrowed_balance == pytest.approx(2_000.0)
    assert borrow_status.core.financing_pnl == pytest.approx(-(2_000.0 * 0.06 / 252.0))
    assert cash_status.cash_balance == pytest.approx(2_000.0)
    assert cash_status.core.financing_pnl == pytest.approx(2_000.0 * 0.02 / 252.0)


def test_margin_account_financing_supports_date_indexed_rate_series():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    cash_rates = pd.Series([0.01, 0.03], index=idx)
    borrow_rates = pd.Series([0.04, 0.08], index=idx)
    policy = MarginPolicy(
        apply_financing=True,
        cash_rate_annual=cash_rates,
        borrow_rate_annual=borrow_rates,
        trading_days_per_year=252,
    )
    account = MarginAccount(policy)

    borrow_day1 = account.evaluate(
        equity=8_000.0,
        initial_margin_requirement=10_000.0,
        open_contracts=1,
        as_of=idx[0],
    )
    borrow_day2 = account.evaluate(
        equity=8_000.0,
        initial_margin_requirement=10_000.0,
        open_contracts=1,
        as_of=idx[1],
    )

    assert borrow_day1.core.financing_pnl == pytest.approx(-(2_000.0 * 0.04 / 252.0))
    assert borrow_day2.core.financing_pnl == pytest.approx(-(2_000.0 * 0.08 / 252.0))
