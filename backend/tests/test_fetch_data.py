import pytest
import pandas as pd
from backend.pipeline.fetch_data import fetch_ohlcv, fetch_fred_series, TICKER_UNIVERSE


def test_ticker_universe_not_empty():
    assert len(TICKER_UNIVERSE) >= 50
    assert "AAPL" in [t["symbol"] for t in TICKER_UNIVERSE]


def test_fetch_ohlcv_returns_dataframe():
    df = fetch_ohlcv("AAPL", period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)
    assert len(df) > 0
    assert df.index.is_monotonic_increasing


def test_fetch_ohlcv_invalid_ticker():
    df = fetch_ohlcv("ZZZZZNOTREAL", period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fetch_fred_series_returns_dataframe():
    # This test will pass with empty DataFrame if no FRED_API_KEY is set
    df = fetch_fred_series("DFF", observation_start="2024-01-01")
    assert isinstance(df, pd.DataFrame)
    assert "value" in df.columns
