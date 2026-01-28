"""Trend detection and signal generation from moving average crossovers."""

import pandas as pd

# Signal values
BULLISH = 1
NEUTRAL = 0
BEARISH = -1


def _calculate_crossover_signal(
    series_fast: pd.Series, series_slow: pd.Series
) -> pd.Series:
    """Calculate crossover signal between two series."""
    signal = pd.Series(NEUTRAL, index=series_fast.index)
    signal[series_fast > series_slow] = BULLISH
    signal[series_fast < series_slow] = BEARISH
    return signal


def analyze_trends(
    data: pd.DataFrame,
    ma_short: int = 50,
    ma_long: int = 200,
) -> pd.DataFrame:
    """Generate MA_Signal, ST_Signal, Price_MA_Signal, and a composite Signal column.
    Values: 1 = bullish, 0 = neutral, -1 = bearish.
    """
    print("Analyzing trends...")

    ma_short_col = f"MA{ma_short}"
    ma_long_col = f"MA{ma_long}"

    data["MA_Signal"] = _calculate_crossover_signal(data[ma_short_col], data[ma_long_col])
    data["ST_Signal"] = _calculate_crossover_signal(data["EMA10"], data["EMA30"])
    data["Price_MA_Signal"] = _calculate_crossover_signal(data["Close"], data[ma_short_col])

    # Composite: MA signal as base, overridden to bearish when short-term or price signals disagree
    data["Signal"] = data["MA_Signal"].copy()
    bearish_override = (data["Price_MA_Signal"] == BEARISH) | (data["ST_Signal"] == BEARISH)
    data.loc[bearish_override, "Signal"] = BEARISH

    data["Signal_Change"] = data["Signal"].diff()

    print("Trends analyzed")
    return data
