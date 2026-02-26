import pytest
import pandas as pd
import numpy as np
from backend.ml.dataset import build_time_series_dataset
from backend.ml.model import create_tft_model, TFTConfig


def _make_feature_df(n_tickers: int = 3, n_days: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    symbols = [f"SYM{i}" for i in range(n_tickers)]
    dfs = []
    for sym in symbols:
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        df = pd.DataFrame({
            "Close": close,
            "Open": close + 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Volume": np.random.randint(1e6, 5e7, n_days).astype(float),
            "RSI_14": np.random.uniform(20, 80, n_days),
            "MACD_12_26_9": np.random.randn(n_days),
            "ATR_14": np.random.uniform(0.5, 3, n_days),
            "return_lag_1": np.random.randn(n_days) * 0.01,
            "return_lag_5": np.random.randn(n_days) * 0.02,
            "symbol": sym,
            "time_idx": range(n_days),
        }, index=dates)
        dfs.append(df)
    return pd.concat(dfs).reset_index(names="date")


def test_build_dataset_returns_training_and_validation():
    df = _make_feature_df()
    training, validation = build_time_series_dataset(df, max_prediction_length=20)
    assert len(training) > 0
    assert len(validation) > 0


def test_create_tft_model_returns_model():
    df = _make_feature_df()
    training, _ = build_time_series_dataset(df, max_prediction_length=20)
    config = TFTConfig(learning_rate=0.01, hidden_size=16, attention_head_size=1)
    model = create_tft_model(training, config)
    assert model is not None
