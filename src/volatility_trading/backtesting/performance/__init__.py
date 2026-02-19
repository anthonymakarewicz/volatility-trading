"""Performance metric calculators, tables, and console formatters."""

from .calculators import compute_performance_metrics
from .console import (
    format_performance_report,
    format_stressed_risk_report,
    print_performance_report,
    print_stressed_risk_metrics,
)
from .schemas import (
    DrawdownMetrics,
    PerformanceMetricsBundle,
    ReturnMetrics,
    TailMetrics,
    TradeMetrics,
)
from .tables import summarize_by_contracts

__all__ = [
    "ReturnMetrics",
    "DrawdownMetrics",
    "TailMetrics",
    "TradeMetrics",
    "PerformanceMetricsBundle",
    "compute_performance_metrics",
    "summarize_by_contracts",
    "format_performance_report",
    "format_stressed_risk_report",
    "print_performance_report",
    "print_stressed_risk_metrics",
]
