import pytest
import pandas as pd
import numpy as np
from backend.ml.features import compute_technical_indicators, prepare_features


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + abs(np.random.randn(n) * 0.5),
        "Low": close - abs(np.random.randn(n) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)


def test_compute_technical_indicators_adds_columns():
    df = _make_ohlcv()
    result = compute_technical_indicators(df)
    expected_cols = ["RSI_14", "MACD_12_26_9", "BBL_20_2.0", "BBU_20_2.0", "ATR_14"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_technical_indicators_no_future_leak():
    df = _make_ohlcv(200)
    result = compute_technical_indicators(df)
    partial = compute_technical_indicators(df.iloc[:150])
    # Row 149 should be identical in both
    assert abs(result.iloc[149]["RSI_14"] - partial.iloc[149]["RSI_14"]) < 1e-10


def test_prepare_features_returns_clean_dataframe():
    df = _make_ohlcv(200)
    result = prepare_features(df, symbol="AAPL")
    assert "symbol" in result.columns
    assert not result.isnull().any().any(), "No NaN values allowed in output"
    assert len(result) < len(df), "Warmup rows should be dropped"
