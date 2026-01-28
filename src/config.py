"""Configuration settings for stock analysis."""

# Stock Settings
SYMBOL: str = "AAPL"
START_DATE: str = "2023-06-01"
END_DATE: str = "2026-01-28"
DISPLAY_START: str = "2024-06-01"

# Technical Indicator Settings
MA_SHORT: int = 50
MA_LONG: int = 200
RSI_PERIOD: int = 14
BOLLINGER_PERIOD: int = 20
BOLLINGER_STD: float = 2.0

# Data Fetching Settings
MAX_RETRIES: int = 3
RETRY_DELAY: int = 10  # seconds

# Output Settings
CHART_DPI: int = 300
SAVE_CHART: bool = True
SHOW_CHART: bool = False

# Machine Learning Settings
ENABLE_ML_PREDICTION: bool = True
ML_LOOKBACK_DAYS: int = 90
ML_EPOCHS: int = 200
ML_VERBOSE: int = 1  # 0=silent, 1=progress bar
