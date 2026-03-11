import pandas as pd

from volatility_trading.signals import LongOnlySignal, ZScoreSignal
from volatility_trading.signals._wrappers import InvertedSignal


def test_inverted_signal_swaps_long_and_short_entries():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    signal = InvertedSignal(LongOnlySignal())

    signals = signal.generate_signals(pd.Series([1.0, 2.0, 3.0], index=idx))

    assert list(signals["long"]) == [False, False, False]
    assert list(signals["short"]) == [True, True, True]
    assert list(signals["exit"]) == [False, False, False]


def test_inverted_signal_delegates_params_and_preserves_zscore():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    signal = InvertedSignal(ZScoreSignal(window=2, entry=1.0, exit=0.5))
    signal.set_params(entry=1.5)

    signals = signal.generate_signals(pd.Series([1.0, 2.0, 5.0, 3.0], index=idx))

    assert signal.get_params()["strategy__entry"] == 1.5
    assert list(signals["long"]) == [False, False, True, False]
    assert list(signals["short"]) == [False, False, False, False]
    assert list(signals["exit"]) == [False, False, False, True]
    assert signal.get_z_score() is not None
