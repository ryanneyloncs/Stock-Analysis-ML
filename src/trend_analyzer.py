"""
Trend analysis module for detecting market trends and generating signals.

Analyzes technical indicators to determine market trends and generate
trading signals based on moving average crossovers.
"""

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
    """
    Analyze market trends and generate trading signals.

    Generates multiple signal types:
    - MA_Signal: Based on short/long MA crossover (Golden Cross/Death Cross)
    - ST_Signal: Short-term signal based on EMA10/EMA30 crossover
    - Price_MA_Signal: Price position relative to short-term MA
    - Signal: Composite signal combining all indicators

    Signal values:
    - 1: Bullish (uptrend)
    - 0: Neutral
    - -1: Bearish (downtrend)

    Args:
        data: Stock data with calculated technical indicators.
        ma_short: Short-term moving average period.
        ma_long: Long-term moving average period.

    Returns:
        DataFrame with trend signals added as new columns.
    """
    print("Analyzing trends...")

    ma_short_col = f"MA{ma_short}"
    ma_long_col = f"MA{ma_long}"

    # MA Signal: Golden Cross (bullish) / Death Cross (bearish)
    data["MA_Signal"] = _calculate_crossover_signal(data[ma_short_col], data[ma_long_col])

    # Short-term Signal: EMA10 vs EMA30
    data["ST_Signal"] = _calculate_crossover_signal(data["EMA10"], data["EMA30"])

    # Price vs MA Signal: Price position relative to short MA
    data["Price_MA_Signal"] = _calculate_crossover_signal(data["Close"], data[ma_short_col])

    # Composite Signal: Start with MA signal, override to bearish if other signals are bearish
    data["Signal"] = data["MA_Signal"].copy()
    bearish_override = (data["Price_MA_Signal"] == BEARISH) | (data["ST_Signal"] == BEARISH)
    data.loc[bearish_override, "Signal"] = BEARISH

    # Track signal changes for identifying trend reversals
    data["Signal_Change"] = data["Signal"].diff()

    print("Trends analyzed")
    return data
