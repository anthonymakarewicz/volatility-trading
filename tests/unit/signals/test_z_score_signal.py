import pandas as pd
import pytest

from volatility_trading.signals.z_score_signal import ZScoreSignal, compute_zscore


def test_compute_zscore_uses_only_prior_observations():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    z_score = compute_zscore(series, window=2)

    assert pd.isna(z_score.iloc[0])
    assert pd.isna(z_score.iloc[1])
    assert z_score.iloc[2] == pytest.approx(2.1213203435596424)
    assert z_score.iloc[3] == pytest.approx(2.1213203435596424)


def test_compute_zscore_ignores_isolated_missing_observations_in_history():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series([1.0, 2.0, 3.0, float("nan"), 4.0], index=idx)

    z_score = compute_zscore(series, window=2)

    assert pd.isna(z_score.iloc[0])
    assert pd.isna(z_score.iloc[1])
    assert z_score.iloc[2] == pytest.approx(2.1213203435596424)
    assert pd.isna(z_score.iloc[3])
    assert z_score.iloc[4] == pytest.approx(2.1213203435596424)


def test_z_score_signal_short_entry_and_exit_on_mean_reversion():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    series = pd.Series([1.0, 2.0, 5.0, 3.0], index=idx)
    signal = ZScoreSignal(window=2, entry=1.0, exit=0.5)

    signals = signal.generate_signals(series)
    z_score = signal.get_z_score()

    assert list(signals["short"]) == [False, False, True, False]
    assert list(signals["long"]) == [False, False, False, False]
    assert list(signals["exit"]) == [False, False, False, True]
    assert z_score is not None
    assert z_score.iloc[2] > 1.0


def test_z_score_signal_recovers_after_isolated_missing_observation():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series([1.0, 2.0, 5.0, float("nan"), 3.0], index=idx)
    signal = ZScoreSignal(window=2, entry=1.0, exit=0.5)

    signals = signal.generate_signals(series)
    z_score = signal.get_z_score()

    assert list(signals["short"]) == [False, False, True, False, False]
    assert list(signals["long"]) == [False, False, False, False, False]
    assert list(signals["exit"]) == [False, False, False, False, True]
    assert z_score is not None
    assert pd.isna(z_score.iloc[3])
    assert z_score.iloc[4] == pytest.approx(-0.23570226039551587)


def test_z_score_signal_long_entry_and_exit_on_mean_reversion():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    series = pd.Series([5.0, 4.0, 1.0, 3.0], index=idx)
    signal = ZScoreSignal(window=2, entry=1.0, exit=0.5)

    signals = signal.generate_signals(series)
    z_score = signal.get_z_score()

    assert list(signals["long"]) == [False, False, True, False]
    assert list(signals["short"]) == [False, False, False, False]
    assert list(signals["exit"]) == [False, False, False, True]
    assert z_score is not None
    assert z_score.iloc[2] < -1.0
