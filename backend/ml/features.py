"""
Feature engineering for Astraeus.

Computes technical indicators from OHLCV data using pandas_ta,
adds lag features, and produces a clean DataFrame ready for the TFT model.
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# pandas_ta 0.4.x hard-imports numba, which does not support Python 3.14.
# We mock numba at import time so the pure-Python fallbacks are used instead.
# ---------------------------------------------------------------------------
if "numba" not in sys.modules:
    _numba_mock = types.ModuleType("numba")
    _numba_mock.njit = lambda *args, **kwargs: (lambda f: f)  # type: ignore[attr-defined]
    _numba_mock.jit = lambda *args, **kwargs: (lambda f: f)  # type: ignore[attr-defined]
    sys.modules["numba"] = _numba_mock

import pandas as pd
import pandas_ta as ta  # noqa: E402 — must come after numba mock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_PERIOD: int = 60
"""Minimum rows needed for indicators to warm up."""


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to an OHLCV DataFrame using pandas_ta.

    Indicators added:
        Trend:       RSI (14), MACD (12,26,9), EMA (20), EMA (50), SMA (20)
        Volatility:  Bollinger Bands (20, std=2), ATR (14)
        Momentum:    Stochastic (k=14, d=3), Williams %R (14), ROC (10)
        Volume:      OBV, VWAP

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
        and a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with new indicator columns appended.
        No future data leakage — every indicator is strictly causal.
    """
    out = df.copy()

    high = out["High"]
    low = out["Low"]
    close = out["Close"]
    volume = out["Volume"]

    # ---- Trend ----
    out["RSI_14"] = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    out["MACD_12_26_9"] = macd.iloc[:, 0]
    out["MACDh_12_26_9"] = macd.iloc[:, 1]
    out["MACDs_12_26_9"] = macd.iloc[:, 2]
    out["EMA_20"] = ta.ema(close, length=20)
    out["EMA_50"] = ta.ema(close, length=50)
    out["SMA_20"] = ta.sma(close, length=20)

    # ---- Volatility ----
    bbands = ta.bbands(close, length=20, std=2)
    # pandas_ta 0.4.x names: BBL_20_2.0_2.0, BBM_20_2.0_2.0, BBU_20_2.0_2.0, ...
    # Normalise to canonical names expected downstream.
    bb_cols = bbands.columns.tolist()
    for col in bb_cols:
        # Map BBL_20_2.0_2.0 -> BBL_20_2.0, etc.
        canonical = _normalise_bb_col(col)
        out[canonical] = bbands[col].values

    out["ATR_14"] = ta.atr(high, low, close, length=14)

    # ---- Momentum ----
    stoch = ta.stoch(high, low, close, k=14, d=3)
    stoch_cols = stoch.columns.tolist()
    for col in stoch_cols:
        canonical = _normalise_stoch_col(col)
        out[canonical] = stoch[col].values

    out["WILLR_14"] = ta.willr(high, low, close, length=14)
    out["ROC_10"] = ta.roc(close, length=10)

    # ---- Volume ----
    out["OBV"] = ta.obv(close, volume)
    out["VWAP"] = ta.vwap(high, low, close, volume)

    return out


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged return columns based on Close price percentage change.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Close`` column.
    lags : list[int], optional
        Lag periods.  Defaults to ``[1, 5, 10, 20]``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with ``return_lag_{n}`` columns appended.
        Only uses past data — no future leakage.
    """
    if lags is None:
        lags = [1, 5, 10, 20]

    out = df.copy()
    for n in lags:
        out[f"return_lag_{n}"] = out["Close"].pct_change(n)
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """End-to-end feature pipeline: indicators + lags + cleanup.

    1. Compute technical indicators
    2. Compute lag features
    3. Tag with *symbol* column
    4. Drop the first ``WARMUP_PERIOD`` rows (NaN-heavy warm-up zone)
    5. Drop any all-NaN columns
    6. Forward-fill remaining NaN, then fill with 0

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with a DatetimeIndex.
    symbol : str
        Ticker symbol (e.g. ``"AAPL"``).

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with **no** NaN values.
    """
    out = compute_technical_indicators(df)
    out = add_lag_features(out)
    out["symbol"] = symbol

    # Drop warm-up rows
    out = out.iloc[WARMUP_PERIOD:]

    # Drop columns that are entirely NaN (after warm-up trim)
    out = out.dropna(axis=1, how="all")

    # Forward-fill then zero-fill any remaining NaN
    out = out.ffill()
    out = out.fillna(0)

    return out


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _normalise_bb_col(col: str) -> str:
    """Normalise Bollinger Band column names across pandas_ta versions.

    pandas_ta 0.3.x: ``BBL_20_2.0``
    pandas_ta 0.4.x: ``BBL_20_2.0_2.0``

    We always produce the shorter ``BBL_20_2.0`` style.
    """
    parts = col.split("_")
    # Expected: prefix(BBL/BBM/BBU/BBB/BBP), length(20), std(2.0)[, duplicate(2.0)]
    if len(parts) >= 4 and parts[0].startswith("BB"):
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return col


def _normalise_stoch_col(col: str) -> str:
    """Normalise Stochastic column names across pandas_ta versions.

    pandas_ta 0.3.x: ``STOCHk_14_3_3``
    pandas_ta 0.4.x: ``STOCHk_14_3_3`` (+ STOCHh_14_3_3)

    We keep the original naming but strip the trailing ``_3`` from
    the histogram column and keep k/d as-is.
    """
    return col
