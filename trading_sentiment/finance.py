"""
data_ingestion/finance.py
~~~~~~~~~~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Fetches OHLCV price data from Yahoo Finance (yfinance) for a given ticker.
Used by the dashboard and backtesting modules to correlate price movement
with sentiment indices.
"""

import logging
from datetime import datetime
from typing import Iterator

import yfinance as yf
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Supported intervals and their max allowed periods ─────────────────────────
# yfinance enforces limits; this guard prevents silent empty DataFrames.
_MAX_PERIOD_FOR_INTERVAL: dict[str, list[str]] = {
    "1m":  ["1d", "2d", "5d", "7d"],
    "2m":  ["1d", "5d", "7d", "60d"],
    "5m":  ["1d", "5d", "7d", "60d"],
    "15m": ["1d", "5d", "7d", "60d"],
    "30m": ["1d", "5d", "7d", "60d"],
    "60m": ["1d", "5d", "7d", "60d", "730d"],
    "1h":  ["1d", "5d", "7d", "60d", "730d"],
    "1d":  ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
}


def fetch_financial_data(
    ticker:   str = "AAPL",
    period:   str = "1d",
    interval: str = "5m",
) -> Iterator[dict]:
    """
    Download OHLCV data for *ticker* and yield each bar as a dictionary.

    Args:
        ticker:   Yahoo Finance ticker symbol (e.g. 'AAPL', 'BTC-USD').
        period:   Lookback period (e.g. '1d', '5d', '1mo').
        interval: Bar interval (e.g. '1m', '5m', '1h', '1d').

    Yields:
        dict with keys: datetime, open, high, low, close, volume.

    Raises:
        ValueError: Ticker is empty or data download returned no rows.
    """
    if not ticker:
        raise ValueError("Ticker symbol must not be empty.")

    logger.info("Downloading %s data for %s (interval=%s)…", period, ticker, interval)
    data: pd.DataFrame = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,   # adjust for splits/dividends automatically
    )

    if data.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"(period='{period}', interval='{interval}'). "
            "Check that the ticker and time range are valid."
        )

    logger.info("Downloaded %d bar(s) for %s.", len(data), ticker)

    for idx, row in data.iterrows():
        yield {
            "datetime": idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
            "open":     float(row["Open"]),
            "high":     float(row["High"]),
            "low":      float(row["Low"]),
            "close":    float(row["Close"]),
            "volume":   int(row["Volume"]),
        }


def fetch_latest_price(ticker: str = "AAPL") -> dict:
    """
    Return only the most recent OHLCV bar for *ticker*.

    Args:
        ticker: Yahoo Finance ticker symbol.

    Returns:
        dict with keys: datetime, open, high, low, close, volume.
    """
    bars = list(fetch_financial_data(ticker, period="1d", interval="1m"))
    if not bars:
        raise ValueError(f"No recent data found for '{ticker}'.")
    return bars[-1]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for bar in fetch_financial_data("AAPL", period="1d", interval="5m"):
        print(bar)
