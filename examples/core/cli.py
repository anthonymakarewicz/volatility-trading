"""Shared CLI contracts for executable examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CommonExampleConfig:
    """Common runtime arguments shared by all VRP example scripts."""

    ticker: str
    start: str
    end: str
    initial_capital: float
    commission_per_leg: float
    hedge_fee_bps: float


@dataclass(frozen=True)
class VrpExampleConfig(CommonExampleConfig):
    """Extended runtime arguments used by the full end-to-end VRP script."""

    rebalance_period: int
    risk_budget_pct: float
    margin_budget_pct: float


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ticker", default="SPY", help="Underlying ticker.")
    parser.add_argument(
        "--start",
        default="2011-01-01",
        help="Backtest start date (YYYY or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2017-12-31",
        help="Backtest end date (YYYY or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=50_000.0,
        help="Initial capital in USD.",
    )
    parser.add_argument(
        "--commission-per-leg",
        type=float,
        default=0.0,
        help="Commission paid per option leg.",
    )
    parser.add_argument(
        "--hedge-fee-bps",
        type=float,
        default=1.0,
        help="Hedge fee in basis points (1.0 means 1 bps).",
    )
    return parser


def parse_common_args(description: str) -> CommonExampleConfig:
    """Parse common example CLI args."""
    args = _common_parser(description).parse_args()
    return CommonExampleConfig(
        ticker=str(args.ticker).strip().upper(),
        start=str(args.start),
        end=str(args.end),
        initial_capital=float(args.initial_capital),
        commission_per_leg=float(args.commission_per_leg),
        hedge_fee_bps=float(args.hedge_fee_bps),
    )


def parse_vrp_args(description: str) -> VrpExampleConfig:
    """Parse end-to-end VRP script args."""
    parser = _common_parser(description)
    parser.add_argument(
        "--rebalance-period",
        type=int,
        default=10,
        help="Rebalance period in calendar days.",
    )
    parser.add_argument(
        "--risk-budget-pct",
        type=float,
        default=1.0,
        help="Risk budget as fraction of equity (0..1+).",
    )
    parser.add_argument(
        "--margin-budget-pct",
        type=float,
        default=0.4,
        help="Initial margin budget as fraction of equity (0..1).",
    )
    args = parser.parse_args()
    return VrpExampleConfig(
        ticker=str(args.ticker).strip().upper(),
        start=str(args.start),
        end=str(args.end),
        initial_capital=float(args.initial_capital),
        commission_per_leg=float(args.commission_per_leg),
        hedge_fee_bps=float(args.hedge_fee_bps),
        rebalance_period=int(args.rebalance_period),
        risk_budget_pct=float(args.risk_budget_pct),
        margin_budget_pct=float(args.margin_budget_pct),
    )
