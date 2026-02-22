from dataclasses import dataclass

import pandas as pd

from volatility_trading.strategies.options_core import (
    ExitRuleSet,
    SameDayReentryPolicy,
)


@dataclass
class _DummyPosition:
    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None


def test_period_exit_rules_rebalance_only():
    rule_set = ExitRuleSet.period_rules()
    position = _DummyPosition(
        rebalance_date=pd.Timestamp("2020-01-10"),
        max_hold_date=pd.Timestamp("2020-01-20"),
    )
    exit_type = rule_set.evaluate(
        curr_date=pd.Timestamp("2020-01-10"),
        position=position,
    )
    assert exit_type == "Rebalance Period"


def test_period_exit_rules_combined():
    rule_set = ExitRuleSet.period_rules()
    position = _DummyPosition(
        rebalance_date=pd.Timestamp("2020-01-10"),
        max_hold_date=pd.Timestamp("2020-01-10"),
    )
    exit_type = rule_set.evaluate(
        curr_date=pd.Timestamp("2020-01-10"),
        position=position,
    )
    assert exit_type == "Rebalance/Max Holding Period"


def test_same_day_reentry_policy_by_exit_type():
    policy = SameDayReentryPolicy(
        allow_on_rebalance=True,
        allow_on_max_holding=False,
    )
    assert policy.allows("Rebalance Period") is True
    assert policy.allows("Max Holding Period") is False
    assert policy.allows("Rebalance/Max Holding Period") is True
    assert policy.allows("Margin Call Liquidation") is False


def test_reentry_policy_from_trade_rows():
    policy = SameDayReentryPolicy(
        allow_on_rebalance=False,
        allow_on_max_holding=False,
        allow_on_margin_liquidation=False,
        allow_on_partial_margin_liquidation=True,
    )
    rows = [
        {"exit_type": "Margin Call Partial Liquidation"},
    ]
    assert policy.allow_from_trade_rows(rows) is True
