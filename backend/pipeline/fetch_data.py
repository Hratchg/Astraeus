"""
Data fetching layer for Astraeus.

Pulls OHLCV stock data via yfinance and macroeconomic series via FRED API.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker universe: 25 Technology + 25 Energy = 50 stocks
# ---------------------------------------------------------------------------

TICKER_UNIVERSE: list[dict[str, str]] = [
    # ---- Technology (25) ----
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Technology"},
    {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "sector": "Technology"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technology"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "sector": "Technology"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
    {"symbol": "QCOM", "name": "Qualcomm Incorporated", "sector": "Technology"},
    {"symbol": "TXN", "name": "Texas Instruments Incorporated", "sector": "Technology"},
    {"symbol": "NOW", "name": "ServiceNow Inc.", "sector": "Technology"},
    {"symbol": "PANW", "name": "Palo Alto Networks Inc.", "sector": "Technology"},
    {"symbol": "SNPS", "name": "Synopsys Inc.", "sector": "Technology"},
    {"symbol": "CDNS", "name": "Cadence Design Systems Inc.", "sector": "Technology"},
    {"symbol": "MU", "name": "Micron Technology Inc.", "sector": "Technology"},
    {"symbol": "MRVL", "name": "Marvell Technology Inc.", "sector": "Technology"},
    {"symbol": "LRCX", "name": "Lam Research Corporation", "sector": "Technology"},
    {"symbol": "AMAT", "name": "Applied Materials Inc.", "sector": "Technology"},
    {"symbol": "KLAC", "name": "KLA Corporation", "sector": "Technology"},
    # ---- Energy (25) ----
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy"},
    {"symbol": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
    {"symbol": "COP", "name": "ConocoPhillips", "sector": "Energy"},
    {"symbol": "SLB", "name": "Schlumberger Limited", "sector": "Energy"},
    {"symbol": "EOG", "name": "EOG Resources Inc.", "sector": "Energy"},
    {"symbol": "MPC", "name": "Marathon Petroleum Corporation", "sector": "Energy"},
    {"symbol": "PSX", "name": "Phillips 66", "sector": "Energy"},
    {"symbol": "VLO", "name": "Valero Energy Corporation", "sector": "Energy"},
    {"symbol": "PXD", "name": "Pioneer Natural Resources Company", "sector": "Energy"},
    {"symbol": "OXY", "name": "Occidental Petroleum Corporation", "sector": "Energy"},
    {"symbol": "WMB", "name": "Williams Companies Inc.", "sector": "Energy"},
    {"symbol": "HES", "name": "Hess Corporation", "sector": "Energy"},
    {"symbol": "DVN", "name": "Devon Energy Corporation", "sector": "Energy"},
    {"symbol": "HAL", "name": "Halliburton Company", "sector": "Energy"},
    {"symbol": "BKR", "name": "Baker Hughes Company", "sector": "Energy"},
    {"symbol": "FANG", "name": "Diamondback Energy Inc.", "sector": "Energy"},
    {"symbol": "KMI", "name": "Kinder Morgan Inc.", "sector": "Energy"},
    {"symbol": "OKE", "name": "ONEOK Inc.", "sector": "Energy"},
    {"symbol": "TRGP", "name": "Targa Resources Corp.", "sector": "Energy"},
    {"symbol": "CTRA", "name": "Coterra Energy Inc.", "sector": "Energy"},
    {"symbol": "EQT", "name": "EQT Corporation", "sector": "Energy"},
    {"symbol": "APA", "name": "APA Corporation", "sector": "Energy"},
    {"symbol": "MRO", "name": "Marathon Oil Corporation", "sector": "Energy"},
    {"symbol": "ENPH", "name": "Enphase Energy Inc.", "sector": "Energy"},
    {"symbol": "CEG", "name": "Constellation Energy Corporation", "sector": "Energy"},
]

# ---------------------------------------------------------------------------
# FRED macroeconomic series
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "DFF": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "VIXCLS": "VIX Index",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
}

# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker via yfinance.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    period : str
        Look-back period accepted by yfinance (e.g. "2y", "1mo").
    interval : str
        Bar interval (e.g. "1d", "1h").

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        tz-naive DatetimeIndex.  Returns an empty DataFrame on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        # Keep only OHLCV columns
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in ohlcv_cols if c in df.columns]]

        # Make index tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception:
        logger.exception("Failed to fetch OHLCV for %s", symbol)
        return pd.DataFrame()


def fetch_fred_series(
    series_id: str,
    observation_start: str = "2020-01-01",
) -> pd.DataFrame:
    """Fetch a FRED macroeconomic series via fredapi.

    Requires ``FRED_API_KEY`` in the environment (or a ``.env`` file).
    Returns an empty DataFrame with a ``value`` column when the key is
    missing or the request fails.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. "DFF").
    observation_start : str
        Start date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``value`` column indexed by date.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.info("FRED_API_KEY not set; returning empty DataFrame")
        return pd.DataFrame(columns=["value"])

    try:
        from fredapi import Fred  # lazy import — only needed when key exists

        fred = Fred(api_key=api_key)
        series: pd.Series = fred.get_series(
            series_id, observation_start=observation_start
        )
        df = series.to_frame(name="value")
        return df

    except Exception:
        logger.exception("Failed to fetch FRED series %s", series_id)
        return pd.DataFrame(columns=["value"])


def fetch_all_ohlcv(period: str = "2y") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for every ticker in ``TICKER_UNIVERSE``.

    Parameters
    ----------
    period : str
        Look-back period passed through to :func:`fetch_ohlcv`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol -> OHLCV DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for entry in TICKER_UNIVERSE:
        symbol = entry["symbol"]
        logger.info("Fetching %s …", symbol)
        results[symbol] = fetch_ohlcv(symbol, period=period)
    return results
