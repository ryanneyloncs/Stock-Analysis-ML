"""
Visualization module for creating stock analysis charts.

Generates multi-panel charts showing price action, volume, RSI, and MACD
indicators with optional ML predictions overlay.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# Chart styling constants
GRID_ALPHA = 0.3
TREND_ALPHA = 0.2
BB_ALPHA = 0.7
BB_LINEWIDTH = 0.8


def _format_price(x: float, pos: int) -> str:
    """Format price values with dollar sign."""
    return f"${x:.2f}"


def _plot_price_panel(
    ax: Axes,
    display_data: pd.DataFrame,
    ma_short_col: str,
    ma_long_col: str,
    ma_short: int,
    ma_long: int,
    ml_predictions: pd.DataFrame | None,
) -> None:
    """Plot the price panel with indicators and trend shading."""
    # Price and moving averages
    ax.plot(display_data.index, display_data["Close"], label="Close Price", color="blue", linewidth=1.5)
    ax.plot(display_data.index, display_data[ma_short_col], label=f"{ma_short}-Day MA", color="orange", linewidth=1)
    ax.plot(display_data.index, display_data[ma_long_col], label=f"{ma_long}-Day MA", color="purple", linewidth=1)

    # Bollinger Bands
    ax.plot(display_data.index, display_data["BB_upper"], "--", label="BB Upper", color="gray", alpha=BB_ALPHA, linewidth=BB_LINEWIDTH)
    ax.plot(display_data.index, display_data["BB_middle"], "-", label="BB Middle", color="gray", alpha=BB_ALPHA, linewidth=BB_LINEWIDTH)
    ax.plot(display_data.index, display_data["BB_lower"], "--", label="BB Lower", color="gray", alpha=BB_ALPHA, linewidth=BB_LINEWIDTH)

    # ML predictions overlay
    if ml_predictions is not None and not ml_predictions.empty:
        ml_display = ml_predictions[ml_predictions.index >= display_data.index[0]]
        if not ml_display.empty:
            ax.plot(ml_display.index, ml_display["ML_Prediction"], label="ML Prediction", color="red", linewidth=2, linestyle="--", alpha=0.8)

    # Trend shading
    valid_mask = display_data[ma_short_col].notna() & display_data[ma_long_col].notna()
    y_max = display_data["Close"].max() * 1.1

    uptrend = (display_data["Signal"] == 1) & valid_mask
    if uptrend.any():
        ax.fill_between(display_data.index, 0, y_max, where=uptrend, color="lightgreen", alpha=TREND_ALPHA, label="Uptrend")

    downtrend = (display_data["Signal"] == -1) & valid_mask
    if downtrend.any():
        ax.fill_between(display_data.index, 0, y_max, where=downtrend, color="lightcoral", alpha=TREND_ALPHA, label="Downtrend")

    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_price))


def _plot_volume_panel(ax: Axes, display_data: pd.DataFrame) -> None:
    """Plot the volume panel."""
    ax.bar(display_data.index, display_data["Volume"], color="blue", alpha=0.5, label="Volume")
    ax.plot(display_data.index, display_data["Volume_MA20"], color="red", label="20-Day Volume MA", linewidth=1)
    ax.set_ylabel("Volume", fontsize=12)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="upper left")


def _plot_rsi_panel(ax: Axes, display_data: pd.DataFrame) -> None:
    """Plot the RSI panel with overbought/oversold zones."""
    ax.plot(display_data.index, display_data["RSI"], color="purple", label="RSI")
    ax.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax.set_ylabel("RSI", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="upper left")


def _plot_macd_panel(ax: Axes, display_data: pd.DataFrame) -> None:
    """Plot the MACD panel with histogram."""
    ax.plot(display_data.index, display_data["MACD"], label="MACD", color="blue")
    ax.plot(display_data.index, display_data["MACD_signal"], label="Signal Line", color="red")

    # Color-coded histogram
    positive = display_data["MACD_hist"] >= 0
    ax.bar(display_data.index[positive], display_data["MACD_hist"][positive], color="green", width=1, alpha=0.5)
    ax.bar(display_data.index[~positive], display_data["MACD_hist"][~positive], color="red", width=1, alpha=0.5)

    ax.set_ylabel("MACD", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="upper left")


def create_chart(
    data: pd.DataFrame,
    display_data: pd.DataFrame,
    symbol: str,
    ma_short: int = 50,
    ma_long: int = 200,
    dpi: int = 300,
    save: bool = True,
    show: bool = False,
    ml_predictions: pd.DataFrame | None = None,
) -> Figure:
    """
    Create a comprehensive multi-panel stock analysis chart.

    Generates a 4-panel chart with:
    1. Price chart with MAs, Bollinger Bands, trend shading, and ML predictions
    2. Volume chart with 20-day moving average
    3. RSI indicator with overbought/oversold zones
    4. MACD with signal line and histogram

    Args:
        data: Full stock data with all indicators.
        display_data: Filtered data for the display period.
        symbol: Stock ticker symbol for the chart title.
        ma_short: Short-term moving average period.
        ma_long: Long-term moving average period.
        dpi: Chart resolution in dots per inch.
        save: If True, saves chart to {symbol}_analysis.png.
        show: If True, displays the chart in a window.
        ml_predictions: DataFrame with ML predictions for overlay.

    Returns:
        The matplotlib Figure object.
    """
    print("Creating chart...")

    ma_short_col = f"MA{ma_short}"
    ma_long_col = f"MA{ma_long}"

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

    # Panel 1: Price Chart
    ax1 = plt.subplot(gs[0])
    _plot_price_panel(ax1, display_data, ma_short_col, ma_long_col, ma_short, ma_long, ml_predictions)
    ax1.set_title(f"{symbol} Stock Price Analysis", fontsize=14)

    # Panel 2: Volume Chart
    ax2 = plt.subplot(gs[1], sharex=ax1)
    _plot_volume_panel(ax2, display_data)

    # Panel 3: RSI Chart
    ax3 = plt.subplot(gs[2], sharex=ax1)
    _plot_rsi_panel(ax3, display_data)

    # Panel 4: MACD Chart
    ax4 = plt.subplot(gs[3], sharex=ax1)
    _plot_macd_panel(ax4, display_data)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)

    if save:
        output_filename = f"{symbol}_analysis.png"
        plt.savefig(output_filename, dpi=dpi)
        print(f"Chart saved as {output_filename}")

    if show:
        plt.show()

    return fig
