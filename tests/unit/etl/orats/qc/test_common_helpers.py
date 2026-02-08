from __future__ import annotations

import json
from pathlib import Path

from volatility_trading.etl.orats.qc.common_helpers import (
    compute_outcome,
    write_json_reports,
)
from volatility_trading.etl.orats.qc.types import (
    Grade,
    QCCheckResult,
    QCConfig,
    Severity,
)


def test_compute_outcome_counts() -> None:
    results = [
        QCCheckResult(
            name="hard_ok",
            severity=Severity.HARD,
            grade=Grade.OK,
            passed=True,
        ),
        QCCheckResult(
            name="soft_warn",
            severity=Severity.SOFT,
            grade=Grade.WARN,
            passed=True,
        ),
        QCCheckResult(
            name="soft_fail",
            severity=Severity.SOFT,
            grade=Grade.FAIL,
            passed=False,
        ),
        # INFO checks should not affect overall passed/fail.
        QCCheckResult(
            name="info_metric",
            severity=Severity.INFO,
            grade=Grade.FAIL,
            passed=False,
        ),
    ]

    passed, n_hard_fail, n_soft_fail, n_soft_warn = compute_outcome(results)
    assert passed is False
    assert n_hard_fail == 0
    assert n_soft_fail == 1
    assert n_soft_warn == 1


def test_write_json_reports_default_paths(tmp_path: Path) -> None:
    parquet_path = tmp_path / "underlying=AAPL" / "part-0000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_text("dummy", encoding="utf-8")

    results = [
        QCCheckResult(
            name="hard_ok",
            severity=Severity.HARD,
            grade=Grade.OK,
            passed=True,
        )
    ]
    config = QCConfig(ticker="AAPL")

    _parquet_path, out_summary, out_config = write_json_reports(
        write_json=True,
        out_json=None,
        parquet_path=parquet_path,
        results=results,
        config=config,
    )

    assert out_summary is not None
    assert out_config is not None
    assert out_summary.exists()
    assert out_config.exists()

    payload = json.loads(out_summary.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload and payload[0]["name"] == "hard_ok"


def test_write_json_reports_explicit_out_json(tmp_path: Path) -> None:
    out_summary = tmp_path / "qc_summary.json"
    results = [
        QCCheckResult(
            name="soft_warn",
            severity=Severity.SOFT,
            grade=Grade.WARN,
            passed=True,
        )
    ]
    config = QCConfig(ticker="AAPL")

    _parquet_path, summary_path, config_path = write_json_reports(
        write_json=True,
        out_json=out_summary,
        parquet_path=None,
        results=results,
        config=config,
    )

    assert summary_path == out_summary
    assert config_path == out_summary.with_name("qc_config.json")
    assert summary_path.exists()
    assert config_path.exists()
