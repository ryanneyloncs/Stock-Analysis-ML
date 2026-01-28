"""Fetches stock data from Yahoo Finance with retry logic."""

import time

import pandas as pd
import yfinance as yf


def fetch_data_with_retry(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> pd.DataFrame | None:
    """Download OHLCV data with linear backoff retries. Returns None on failure."""
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}: Fetching data for {symbol}...")

        if attempt > 0:
            wait_time = retry_delay * attempt
            print(f"Waiting {wait_time} seconds...")
            time.sleep(wait_time)

        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )

            if not data.empty:
                print(f"Successfully fetched {len(data)} days of data")
                return data

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")

    return None


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns from yfinance and drop rows missing Close."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data = data.dropna(subset=["Close"])

    return data
