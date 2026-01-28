# Stock Analyzer

Technical analysis tool that pulls historical stock data from Yahoo Finance, computes indicators (moving averages, RSI, MACD, Bollinger Bands), and optionally runs an LSTM model to predict next-day closing price.

Outputs a 4-panel chart and a text report with trend stats, returns, volatility, and drawdown.

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python main.py
```

Edit `src/config.py` to change the stock symbol, date range, indicator parameters, or toggle ML prediction on/off.

## Output

- `{SYMBOL}_analysis.png` — chart with price, volume, RSI, and MACD panels
- Console report with performance metrics and (if enabled) ML prediction
