"""
Stock Technical Analysis Tool - Main Entry Point.

Orchestrates the full analysis pipeline:
1. Fetch historical stock data from Yahoo Finance
2. Calculate technical indicators (MA, RSI, MACD, Bollinger Bands)
3. Analyze trends and generate trading signals
4. Train ML model and predict next-day price (optional)
5. Generate visualization and reports

Usage:
    python main.py

Configuration:
    Edit src/config.py to customize analysis parameters.
"""

import sys

import pandas as pd

from src import config
from src.data_fetcher import clean_data, fetch_data_with_retry
from src.indicators import calculate_indicators
from src.ml_predictor import predict_stock_price
from src.reporting import print_ml_report, print_report
from src.trend_analyzer import analyze_trends
from src.visualization import create_chart

SEPARATOR = "=" * 60


def print_header() -> None:
    """Print the application header with stock info."""
    print(SEPARATOR)
    print("STOCK TECHNICAL ANALYSIS TOOL")
    print(SEPARATOR)
    print(f"\nAnalyzing: {config.SYMBOL}")
    print(f"Period: {config.START_DATE} to {config.END_DATE}\n")


def fetch_and_prepare_data() -> pd.DataFrame:
    """Fetch, clean, and calculate indicators for stock data."""
    data = fetch_data_with_retry(
        config.SYMBOL,
        config.START_DATE,
        config.END_DATE,
        config.MAX_RETRIES,
        config.RETRY_DELAY,
    )

    if data is None or data.empty:
        print("Failed to fetch data")
        sys.exit(1)

    data = clean_data(data)
    data = calculate_indicators(
        data,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG,
        rsi_period=config.RSI_PERIOD,
        bollinger_period=config.BOLLINGER_PERIOD,
        bollinger_std=config.BOLLINGER_STD,
    )
    data = analyze_trends(data, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)

    return data


def run_ml_prediction(data: pd.DataFrame) -> tuple:
    """Run ML prediction if enabled. Returns (metrics, next_day, predictions_df)."""
    if not config.ENABLE_ML_PREDICTION:
        return None, None, None

    print(f"\n{SEPARATOR}")
    print("MACHINE LEARNING PREDICTION")
    print(SEPARATOR)

    _, metrics, next_day, predictions_df = predict_stock_price(
        data,
        lookback_days=config.ML_LOOKBACK_DAYS,
        epochs=config.ML_EPOCHS,
        verbose=config.ML_VERBOSE,
    )
    return metrics, next_day, predictions_df


def main() -> None:
    """Main entry point - orchestrates the complete stock analysis pipeline."""
    print_header()

    data = fetch_and_prepare_data()

    display_data = data[data.index >= pd.Timestamp(config.DISPLAY_START)]
    if display_data.empty:
        display_data = data

    ml_metrics, ml_next_day, ml_predictions_df = run_ml_prediction(data)

    create_chart(
        data,
        display_data,
        config.SYMBOL,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG,
        dpi=config.CHART_DPI,
        save=config.SAVE_CHART,
        show=config.SHOW_CHART,
        ml_predictions=ml_predictions_df,
    )

    print_report(
        data,
        display_data,
        config.SYMBOL,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG,
    )

    if ml_metrics is not None:
        print_ml_report(ml_metrics, ml_next_day)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
