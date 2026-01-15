"""
Technical indicators calculation module.

Provides functions to calculate common technical analysis indicators
used in stock market analysis.
"""

import pandas as pd

EPSILON = 1e-10  # Small value to avoid division by zero


def _calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + EPSILON)
    return 100 - (100 / (1 + rs))


def _calculate_bollinger_bands(
    close: pd.Series, period: int, std_multiplier: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (middle, upper, lower)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std_multiplier * std)
    lower = middle - (std_multiplier * std)
    return middle, upper, lower


def _calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, signal line, and histogram."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_indicators(
    data: pd.DataFrame,
    ma_short: int = 50,
    ma_long: int = 200,
    rsi_period: int = 14,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate all technical indicators for the given stock data.

    Adds the following indicators to the DataFrame:
    - Moving Averages (MA{short}, MA{long})
    - RSI (Relative Strength Index)
    - Bollinger Bands (upper, middle, lower)
    - MACD (Moving Average Convergence Divergence)
    - Volume Moving Average
    - Daily Returns
    - EMAs for trend detection

    Args:
        data: Stock data with Close, High, Low, Volume columns.
        ma_short: Short-term moving average period in days.
        ma_long: Long-term moving average period in days.
        rsi_period: RSI calculation period in days.
        bollinger_period: Bollinger Bands period in days.
        bollinger_std: Bollinger Bands standard deviation multiplier.

    Returns:
        DataFrame with all technical indicators added as new columns.
    """
    print("Calculating technical indicators...")

    close = data["Close"]

    # Moving Averages
    data[f"MA{ma_short}"] = close.rolling(window=ma_short).mean()
    data[f"MA{ma_long}"] = close.rolling(window=ma_long).mean()

    # RSI
    data["RSI"] = _calculate_rsi(close, rsi_period)

    # Bollinger Bands
    data["BB_middle"], data["BB_upper"], data["BB_lower"] = _calculate_bollinger_bands(
        close, bollinger_period, bollinger_std
    )

    # MACD
    data["MACD"], data["MACD_signal"], data["MACD_hist"] = _calculate_macd(close)

    # Store intermediate EMAs (used by MACD calculation but also useful for analysis)
    data["EMA12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # Volume Moving Average
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()

    # Daily Returns (percentage)
    data["Daily_Return"] = close.pct_change() * 100

    # EMAs for trend detection
    data["EMA10"] = close.ewm(span=10, adjust=False).mean()
    data["EMA30"] = close.ewm(span=30, adjust=False).mean()

    print("Indicators calculated")
    return data
