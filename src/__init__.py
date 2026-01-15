"""
Stock Technical Analysis Tool - Source Package.

A comprehensive stock analysis toolkit combining traditional technical
analysis with machine learning predictions.

Modules:
    config: Configuration settings for the analyzer
    data_fetcher: Yahoo Finance data retrieval with retry logic
    indicators: Technical indicator calculations (MA, RSI, MACD, etc.)
    trend_analyzer: Trend detection and signal generation
    ml_predictor: LSTM-based price prediction
    visualization: Multi-panel chart generation
    reporting: Text-based analysis reports
"""

__version__ = "1.0.0"
__author__ = "Ryan Neylon"

__all__ = [
    "config",
    "data_fetcher",
    "indicators",
    "trend_analyzer",
    "ml_predictor",
    "visualization",
    "reporting",
]
