"""Text-based report output for trend stats, performance, and ML results."""

from typing import Any

import numpy as np
import pandas as pd

SEPARATOR = "=" * 60
TRADING_DAYS_PER_YEAR = 252


def _get_rsi_status(rsi: float) -> str:
    if rsi > 70:
        return "Overbought"
    if rsi < 30:
        return "Oversold"
    return "Neutral"


def _get_trend_status(signal: int) -> str:
    if signal == 1:
        return "UPTREND"
    if signal == -1:
        return "DOWNTREND"
    return "NEUTRAL"


def _calculate_performance_metrics(display_data: pd.DataFrame) -> dict[str, float]:
    first_close = display_data["Close"].iloc[0]
    last_close = display_data["Close"].iloc[-1]
    total_return = (last_close / first_close - 1) * 100

    daily_returns = display_data["Daily_Return"].dropna()
    annualized_volatility = 0.0
    if len(daily_returns) > 0:
        annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    cumulative_max = display_data["Close"].cummax()
    drawdown = (display_data["Close"] / cumulative_max - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "first_close": first_close,
        "last_close": last_close,
        "total_return": total_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
    }


def print_report(
    data: pd.DataFrame,
    display_data: pd.DataFrame,
    symbol: str,
    ma_short: int = 50,
    ma_long: int = 200,
) -> None:
    """Print trend breakdown, performance metrics, and current indicator readings."""
    ma_short_col = f"MA{ma_short}"
    ma_long_col = f"MA{ma_long}"

    valid_mask = display_data[ma_short_col].notna() & display_data[ma_long_col].notna()
    valid_data = display_data[valid_mask]

    if len(valid_data) == 0:
        print("WARNING: Not enough data for report")
        return

    total_days = len(valid_data)
    uptrend_days = (valid_data["Signal"] == 1).sum()
    downtrend_days = (valid_data["Signal"] == -1).sum()

    metrics = _calculate_performance_metrics(display_data)

    print(f"\n{'=' * 20} {symbol} ANALYSIS REPORT {'=' * 20}")

    print("\n----- TREND ANALYSIS -----")
    print(f"Total trading days: {total_days}")
    print(f"Days in uptrend: {uptrend_days} ({uptrend_days / total_days * 100:.2f}%)")
    print(f"Days in downtrend: {downtrend_days} ({downtrend_days / total_days * 100:.2f}%)")

    print("\n----- PERFORMANCE METRICS -----")
    print(f"Starting price: ${metrics['first_close']:.2f}")
    print(f"Ending price: ${metrics['last_close']:.2f}")
    print(f"Total return: {metrics['total_return']:.2f}%")
    print(f"Annualized volatility: {metrics['annualized_volatility']:.2f}%")
    print(f"Maximum drawdown: {metrics['max_drawdown']:.2f}%")

    print("\n----- TECHNICAL INDICATORS (LATEST) -----")
    latest_rsi = display_data["RSI"].iloc[-1]
    if pd.notna(latest_rsi):
        print(f"RSI: {latest_rsi:.2f} -> {_get_rsi_status(latest_rsi)}")

    current_signal = display_data["Signal"].iloc[-1]
    print(f"\nCurrent trend: {_get_trend_status(current_signal)}")
    print(f"{SEPARATOR}\n")


def print_ml_report(metrics: dict[str, Any] | None, next_day_prediction: float | None) -> None:
    """Print MAE/RMSE scores and the next-day price forecast."""
    if metrics is None or next_day_prediction is None:
        return

    print("\n----- MACHINE LEARNING PREDICTIONS -----")
    print("Model Performance:")
    print(f"  Training MAE: ${metrics['train_mae']:.2f}")
    print(f"  Testing MAE: ${metrics['test_mae']:.2f}")
    print(f"  Training RMSE: ${metrics['train_rmse']:.2f}")
    print(f"  Testing RMSE: ${metrics['test_rmse']:.2f}")
    print(f"\nNext Day Prediction: ${next_day_prediction:.2f}")
    print(f"{SEPARATOR}\n")
