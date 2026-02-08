from __future__ import annotations

from pathlib import Path

import polars as pl

from volatility_trading.etl.orats.qc.types import (
    Grade,
    QCCheckResult,
    Severity,
)


def test_run_options_chain_qc_writes_json_and_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import volatility_trading.etl.orats.qc.options_chain.runner as mod

    df = pl.DataFrame({"x": [1, 2, 3]})

    monkeypatch.setattr(
        mod, "get_parquet_path", lambda proc_root, ticker: tmp_path / "p.parquet"
    )
    monkeypatch.setattr(mod, "read_exercise_style", lambda parquet_path: "EU")
    monkeypatch.setattr(mod, "load_options_chain_df", lambda ticker, proc_root: df)
    monkeypatch.setattr(mod, "apply_roi_filter", lambda df_, **kwargs: df_.head(1))
    monkeypatch.setattr(mod, "log_check", lambda logger, r: None)

    results = [
        QCCheckResult(
            name="hard_ok",
            severity=Severity.HARD,
            grade=Grade.OK,
            passed=True,
        ),
        QCCheckResult(
            name="soft_fail",
            severity=Severity.SOFT,
            grade=Grade.FAIL,
            passed=False,
        ),
    ]
    monkeypatch.setattr(mod, "run_all_checks", lambda **kwargs: results)

    out_json = tmp_path / "qc_summary.json"
    res = mod.run_options_chain_qc(
        ticker="AAPL",
        proc_root=tmp_path,
        out_json=out_json,
        write_json=True,
    )

    assert res.passed is False
    assert res.n_soft_fail == 1
    assert res.out_summary_json == out_json
    assert out_json.exists()
    assert out_json.with_name("qc_config.json").exists()


def test_run_daily_features_qc_writes_json_and_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import volatility_trading.etl.orats.qc.daily_features.runner as mod

    df = pl.DataFrame({"x": [1, 2]})

    monkeypatch.setattr(
        mod, "daily_features_path", lambda proc_root, ticker: tmp_path / "p.parquet"
    )
    monkeypatch.setattr(mod, "read_daily_features", lambda ticker, proc_root: df)
    monkeypatch.setattr(mod, "log_check", lambda logger, r: None)

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
    ]
    monkeypatch.setattr(mod, "run_all_checks", lambda **kwargs: results)

    out_json = tmp_path / "qc_summary.json"
    res = mod.run_daily_features_qc(
        ticker="AAPL",
        proc_root=tmp_path,
        out_json=out_json,
        write_json=True,
    )

    assert res.config.run_roi is False
    assert res.passed is False
    assert res.n_soft_fail == 1
    assert res.out_summary_json == out_json
    assert out_json.exists()
    assert out_json.with_name("qc_config.json").exists()
