# Astraeus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a research-grade financial forecasting dashboard with TFT-based probabilistic predictions, served through a professional fintech-dark Next.js frontend.

**Architecture:** Offline pipeline (fetch data → engineer features → train TFT → export predictions) decoupled from a lightweight FastAPI serving layer. Next.js frontend consumes the API. No GPU needed at serving time.

**Tech Stack:** Python 3.10+ (PyTorch Forecasting, FastAPI, pandas-ta, yfinance, fredapi), Next.js 15 (App Router, Shadcn UI, Tailwind, Lightweight Charts, Recharts)

---

## Prerequisites

- Python 3.10-3.12 virtual environment (PyTorch Forecasting does not support 3.14 yet)
- Node.js 20+
- Git initialized (already done)

---

## Phase 1A: Backend Foundation

### Task 1: Project Scaffolding & Python Environment

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/api/__init__.py`
- Create: `backend/ml/__init__.py`
- Create: `backend/pipeline/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/requirements.txt`
- Create: `backend/predictions/sample/.gitkeep`
- Create: `backend/models/.gitkeep`

**Step 1: Create Python virtual environment**

Run:
```bash
cd C:/Users/King\ Hratch/astraeus
python -m venv .venv --prompt astraeus
# If Python 3.14 fails with PyTorch, use: py -3.12 -m venv .venv --prompt astraeus
```

**Step 2: Create package structure**

Create all `__init__.py` files and directories:

```bash
mkdir -p backend/api/routes backend/ml backend/pipeline backend/tests
mkdir -p backend/predictions/sample backend/models
touch backend/__init__.py backend/api/__init__.py backend/api/routes/__init__.py
touch backend/ml/__init__.py backend/pipeline/__init__.py backend/tests/__init__.py
touch backend/predictions/sample/.gitkeep backend/models/.gitkeep
```

**Step 3: Create requirements.txt**

```
# API
fastapi==0.115.*
uvicorn[standard]==0.34.*

# ML
torch>=2.2.0
pytorch-forecasting>=1.1.0
pytorch-lightning>=2.2.0

# Data
yfinance>=0.2.36
fredapi>=0.5.2
pandas>=2.2.0
pandas-ta>=0.3.14b1

# Utils
python-dotenv>=1.0.0
pydantic>=2.6.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

Write this to `backend/requirements.txt`.

**Step 4: Install dependencies**

Run:
```bash
source .venv/Scripts/activate  # Windows Git Bash
pip install -r backend/requirements.txt
```

Expected: All packages install. If torch fails on Python 3.14, fall back to `py -3.12`.

**Step 5: Verify imports**

Run:
```bash
python -c "import torch; import pytorch_forecasting; import fastapi; print('All imports OK')"
```

Expected: `All imports OK`

**Step 6: Commit**

```bash
git add backend/ .venv  # .venv should be gitignored already
git add backend/requirements.txt backend/__init__.py backend/api/__init__.py
git add backend/api/routes/__init__.py backend/ml/__init__.py
git add backend/pipeline/__init__.py backend/tests/__init__.py
git add backend/predictions/sample/.gitkeep backend/models/.gitkeep
git commit -m "feat: scaffold backend package structure and dependencies"
```

---

### Task 2: Data Fetching Layer

**Files:**
- Create: `backend/pipeline/fetch_data.py`
- Create: `backend/tests/test_fetch_data.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_fetch_data.py
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
    df = fetch_fred_series("DFF", observation_start="2024-01-01")
    assert isinstance(df, pd.DataFrame)
    assert "value" in df.columns
    assert len(df) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_fetch_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.pipeline.fetch_data'`

**Step 3: Write implementation**

```python
# backend/pipeline/fetch_data.py
"""Data fetching from yfinance and FRED API."""
import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

TICKER_UNIVERSE = [
    # Technology (25)
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Technology"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"symbol": "INTC", "name": "Intel Corp.", "sector": "Technology"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "ORCL", "name": "Oracle Corp.", "sector": "Technology"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "CSCO", "name": "Cisco Systems", "sector": "Technology"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
    {"symbol": "QCOM", "name": "Qualcomm Inc.", "sector": "Technology"},
    {"symbol": "TXN", "name": "Texas Instruments", "sector": "Technology"},
    {"symbol": "NOW", "name": "ServiceNow Inc.", "sector": "Technology"},
    {"symbol": "PANW", "name": "Palo Alto Networks", "sector": "Technology"},
    {"symbol": "SNPS", "name": "Synopsys Inc.", "sector": "Technology"},
    {"symbol": "CDNS", "name": "Cadence Design Systems", "sector": "Technology"},
    {"symbol": "MU", "name": "Micron Technology", "sector": "Technology"},
    {"symbol": "MRVL", "name": "Marvell Technology", "sector": "Technology"},
    {"symbol": "LRCX", "name": "Lam Research", "sector": "Technology"},
    {"symbol": "AMAT", "name": "Applied Materials", "sector": "Technology"},
    {"symbol": "KLAC", "name": "KLA Corp.", "sector": "Technology"},
    # Energy (25)
    {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"symbol": "CVX", "name": "Chevron Corp.", "sector": "Energy"},
    {"symbol": "COP", "name": "ConocoPhillips", "sector": "Energy"},
    {"symbol": "SLB", "name": "Schlumberger Ltd.", "sector": "Energy"},
    {"symbol": "EOG", "name": "EOG Resources", "sector": "Energy"},
    {"symbol": "MPC", "name": "Marathon Petroleum", "sector": "Energy"},
    {"symbol": "PSX", "name": "Phillips 66", "sector": "Energy"},
    {"symbol": "VLO", "name": "Valero Energy", "sector": "Energy"},
    {"symbol": "PXD", "name": "Pioneer Natural Resources", "sector": "Energy"},
    {"symbol": "OXY", "name": "Occidental Petroleum", "sector": "Energy"},
    {"symbol": "WMB", "name": "Williams Companies", "sector": "Energy"},
    {"symbol": "HES", "name": "Hess Corp.", "sector": "Energy"},
    {"symbol": "DVN", "name": "Devon Energy", "sector": "Energy"},
    {"symbol": "HAL", "name": "Halliburton Co.", "sector": "Energy"},
    {"symbol": "BKR", "name": "Baker Hughes Co.", "sector": "Energy"},
    {"symbol": "FANG", "name": "Diamondback Energy", "sector": "Energy"},
    {"symbol": "KMI", "name": "Kinder Morgan", "sector": "Energy"},
    {"symbol": "OKE", "name": "ONEOK Inc.", "sector": "Energy"},
    {"symbol": "TRGP", "name": "Targa Resources", "sector": "Energy"},
    {"symbol": "CTRA", "name": "Coterra Energy", "sector": "Energy"},
    {"symbol": "EQT", "name": "EQT Corp.", "sector": "Energy"},
    {"symbol": "APA", "name": "APA Corp.", "sector": "Energy"},
    {"symbol": "MRO", "name": "Marathon Oil Corp.", "sector": "Energy"},
    {"symbol": "ENPH", "name": "Enphase Energy", "sector": "Energy"},
    {"symbol": "CEG", "name": "Constellation Energy", "sector": "Energy"},
]

# FRED series we care about
FRED_SERIES = {
    "DFF": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "VIXCLS": "VIX Index",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
}


def fetch_ohlcv(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance. Returns empty DataFrame on failure."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_fred_series(
    series_id: str,
    observation_start: str = "2020-01-01",
) -> pd.DataFrame:
    """Fetch a FRED series. Returns DataFrame with 'value' column."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        # Fallback: return empty DataFrame if no API key
        return pd.DataFrame(columns=["value"])
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=observation_start)
        df = series.to_frame(name="value")
        df.index = pd.to_datetime(df.index)
        return df.dropna()
    except Exception:
        return pd.DataFrame(columns=["value"])


def fetch_all_ohlcv(period: str = "2y") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all tickers in the universe."""
    results = {}
    for ticker_info in TICKER_UNIVERSE:
        symbol = ticker_info["symbol"]
        df = fetch_ohlcv(symbol, period=period)
        if not df.empty:
            results[symbol] = df
    return results
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_fetch_data.py -v`
Expected: 4 tests pass (test_fetch_fred_series may be skipped if no FRED_API_KEY set — that's OK)

**Step 5: Commit**

```bash
git add backend/pipeline/fetch_data.py backend/tests/test_fetch_data.py
git commit -m "feat: add data fetching layer (yfinance + FRED)"
```

---

### Task 3: Feature Engineering

**Files:**
- Create: `backend/ml/features.py`
- Create: `backend/tests/test_features.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_features.py
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
    # The last row's RSI should only depend on the last 14+ rows, not future data
    partial = compute_technical_indicators(df.iloc[:150])
    # Row 149 should be identical in both
    assert abs(result.iloc[149]["RSI_14"] - partial.iloc[149]["RSI_14"]) < 1e-10


def test_prepare_features_returns_clean_dataframe():
    df = _make_ohlcv(200)
    result = prepare_features(df, symbol="AAPL")
    assert "symbol" in result.columns
    assert not result.isnull().any().any(), "No NaN values allowed in output"
    assert len(result) < len(df), "Warmup rows should be dropped"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# backend/ml/features.py
"""Feature engineering using pandas-ta technical indicators."""
import pandas as pd
import pandas_ta as ta


# Minimum rows needed for all indicators to warm up
WARMUP_PERIOD = 60


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame. No future data leakage."""
    df = df.copy()

    # Trend
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.sma(length=20, append=True)

    # Volatility
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    # Momentum
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.roc(length=10, append=True)

    # Volume
    df.ta.obv(append=True)
    df.ta.vwap(append=True)

    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """Add lagged return features. Only uses past data."""
    df = df.copy()
    for lag in lags:
        df[f"return_lag_{lag}"] = df["Close"].pct_change(lag)
    return df


def prepare_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Full feature pipeline: indicators + lags + cleanup. Returns clean DataFrame."""
    df = compute_technical_indicators(df)
    df = add_lag_features(df)
    df["symbol"] = symbol

    # Drop warmup rows where indicators are NaN
    df = df.iloc[WARMUP_PERIOD:]

    # Drop any remaining NaN columns (VWAP can produce NaN on some data)
    df = df.dropna(axis=1, how="all")
    # Fill any remaining NaN with forward fill then 0
    df = df.ffill().fillna(0)

    return df
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_features.py -v`
Expected: 3 tests pass

**Step 5: Commit**

```bash
git add backend/ml/features.py backend/tests/test_features.py
git commit -m "feat: add feature engineering with pandas-ta indicators"
```

---

### Task 4: TFT Dataset & Model Wrapper

**Files:**
- Create: `backend/ml/dataset.py`
- Create: `backend/ml/model.py`
- Create: `backend/tests/test_model.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_model.py
import pytest
import pandas as pd
import numpy as np
from backend.ml.dataset import build_time_series_dataset
from backend.ml.model import create_tft_model, TFTConfig


def _make_feature_df(n_tickers: int = 3, n_days: int = 200) -> pd.DataFrame:
    """Create a multi-ticker feature DataFrame."""
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write dataset.py**

```python
# backend/ml/dataset.py
"""TimeSeriesDataSet construction for PyTorch Forecasting TFT."""
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


# Features the model will use as time-varying known reals
TIME_VARYING_KNOWN = []  # We don't have future-known features

# Features the model will use as time-varying unknown reals
TIME_VARYING_UNKNOWN = [
    "Close", "Open", "High", "Low", "Volume",
    "RSI_14", "MACD_12_26_9", "ATR_14",
    "return_lag_1", "return_lag_5",
]

MAX_ENCODER_LENGTH = 60  # Look back 60 trading days


def build_time_series_dataset(
    df: pd.DataFrame,
    max_prediction_length: int = 20,
    training_cutoff_frac: float = 0.8,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build training and validation TimeSeriesDataSets from feature DataFrame.

    Args:
        df: Must have columns: symbol, time_idx, and all TIME_VARYING_UNKNOWN columns
        max_prediction_length: Forecast horizon (max 20)
        training_cutoff_frac: Fraction of time_idx used for training

    Returns:
        (training_dataset, validation_dataset)
    """
    # Determine available unknown reals (some indicators may be absent)
    available_unknowns = [c for c in TIME_VARYING_UNKNOWN if c in df.columns]

    max_time_idx = df["time_idx"].max()
    training_cutoff = int(max_time_idx * training_cutoff_frac)

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="Close",
        group_ids=["symbol"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=available_unknowns,
        time_varying_known_reals=TIME_VARYING_KNOWN if TIME_VARYING_KNOWN else None,
        target_normalizer="auto",
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=training_cutoff + 1,
    )

    return training, validation
```

**Step 4: Write model.py**

```python
# backend/ml/model.py
"""TFT model configuration and creation."""
from dataclasses import dataclass
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


@dataclass
class TFTConfig:
    learning_rate: float = 0.001
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    output_size: int = 7  # 7 quantiles


# Quantiles for probabilistic output: gives us 80% and 95% intervals
QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]


def create_tft_model(
    training_dataset: TimeSeriesDataSet,
    config: TFTConfig | None = None,
) -> TemporalFusionTransformer:
    """Create a TFT model from a training dataset."""
    if config is None:
        config = TFTConfig()

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=config.output_size,
        loss=QuantileLoss(quantiles=QUANTILES),
        reduce_on_plateau_patience=4,
    )

    return model
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_model.py -v`
Expected: 2 tests pass

**Step 6: Commit**

```bash
git add backend/ml/dataset.py backend/ml/model.py backend/tests/test_model.py
git commit -m "feat: add TFT dataset builder and model wrapper"
```

---

### Task 5: Explainability Extraction

**Files:**
- Create: `backend/ml/explain.py`
- Create: `backend/tests/test_explain.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_explain.py
import pytest
from backend.ml.explain import extract_feature_importance, extract_temporal_attention


def test_extract_feature_importance_returns_sorted_list():
    # Mock interpretation dict as returned by TFT
    mock_interpretation = {
        "encoder_variables": ["Close", "RSI_14", "Volume", "ATR_14"],
        "encoder_importance": [0.15, 0.30, 0.10, 0.25],
    }
    result = extract_feature_importance(mock_interpretation)
    assert len(result) > 0
    assert result[0]["importance"] >= result[1]["importance"]  # Sorted descending
    assert "feature" in result[0]
    assert "importance" in result[0]


def test_extract_temporal_attention_returns_weights():
    import numpy as np
    mock_attention = np.random.dirichlet(np.ones(60))  # 60 timesteps
    dates = [f"2026-02-{i:02d}" for i in range(1, 61)]
    result = extract_temporal_attention(mock_attention, dates)
    assert len(result) == 60
    assert "date" in result[0]
    assert "weight" in result[0]
    assert abs(sum(r["weight"] for r in result) - 1.0) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_explain.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# backend/ml/explain.py
"""Extract explainability data from trained TFT models."""
import numpy as np


def extract_feature_importance(interpretation: dict) -> list[dict]:
    """Extract and sort variable importance from TFT interpretation.

    Args:
        interpretation: Dict with 'encoder_variables' and 'encoder_importance' keys,
                       as returned by TFT.interpret_output()

    Returns:
        List of {"feature": str, "importance": float} sorted descending by importance.
    """
    variables = interpretation["encoder_variables"]
    importances = interpretation["encoder_importance"]

    if hasattr(importances, "tolist"):
        importances = importances.tolist() if hasattr(importances, "tolist") else list(importances)

    pairs = [
        {"feature": var, "importance": round(float(imp), 4)}
        for var, imp in zip(variables, importances)
    ]
    pairs.sort(key=lambda x: x["importance"], reverse=True)
    return pairs


def extract_temporal_attention(
    attention_weights: np.ndarray,
    dates: list[str],
) -> list[dict]:
    """Convert attention weights array to dated weight list.

    Args:
        attention_weights: 1D array of attention weights (should sum to ~1)
        dates: Corresponding date strings, same length as attention_weights

    Returns:
        List of {"date": str, "weight": float}
    """
    weights = np.array(attention_weights, dtype=float)
    # Normalize to sum to 1
    weights = weights / weights.sum()

    return [
        {"date": date, "weight": round(float(w), 6)}
        for date, w in zip(dates, weights)
    ]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_explain.py -v`
Expected: 2 tests pass

**Step 5: Commit**

```bash
git add backend/ml/explain.py backend/tests/test_explain.py
git commit -m "feat: add explainability extraction (feature importance + temporal attention)"
```

---

### Task 6: Pipeline Orchestrator & Export

**Files:**
- Create: `backend/pipeline/build_features.py`
- Create: `backend/pipeline/train.py`
- Create: `backend/pipeline/predict.py`
- Create: `backend/pipeline/export.py`
- Create: `backend/pipeline/run_pipeline.py`
- Create: `backend/tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_pipeline.py
import pytest
import json
import os
import tempfile
from backend.pipeline.export import export_predictions, export_explainability


def test_export_predictions_writes_valid_json():
    predictions = {
        "symbol": "AAPL",
        "horizon_days": 5,
        "generated_at": "2026-02-25T00:00:00Z",
        "model_version": "tft-v1",
        "historical": [
            {"date": "2026-02-20", "open": 182.1, "high": 184.5, "low": 181.2, "close": 183.7, "volume": 54000000}
        ],
        "predictions": [
            {"date": "2026-02-26", "median": 185.2, "lower_80": 183.1, "upper_80": 187.3, "lower_95": 181.5, "upper_95": 189.0}
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        export_predictions(predictions, output_dir=tmpdir)
        path = os.path.join(tmpdir, "AAPL_h5.json")
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["symbol"] == "AAPL"
        assert loaded["horizon_days"] == 5


def test_export_explainability_writes_valid_json():
    explain_data = {
        "symbol": "AAPL",
        "horizon_days": 5,
        "feature_importance": [{"feature": "RSI_14", "importance": 0.3}],
        "temporal_attention": [{"date": "2026-02-20", "weight": 0.5}],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        export_explainability(explain_data, output_dir=tmpdir)
        path = os.path.join(tmpdir, "AAPL_explain_h5.json")
        assert os.path.exists(path)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write build_features.py**

```python
# backend/pipeline/build_features.py
"""Build features for all tickers."""
import pandas as pd
from backend.ml.features import prepare_features


def build_all_features(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build features for all tickers, assign time_idx, return combined DataFrame."""
    all_dfs = []
    for symbol, df in ohlcv_data.items():
        featured = prepare_features(df, symbol=symbol)
        if len(featured) > 0:
            all_dfs.append(featured)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs).reset_index(names="date")

    # Assign global time_idx based on date
    unique_dates = sorted(combined["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    combined["time_idx"] = combined["date"].map(date_to_idx)

    return combined
```

**Step 4: Write train.py**

```python
# backend/pipeline/train.py
"""Train TFT model."""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from backend.ml.dataset import build_time_series_dataset
from backend.ml.model import create_tft_model, TFTConfig
import pandas as pd


def train_tft(
    feature_df: pd.DataFrame,
    max_prediction_length: int = 20,
    max_epochs: int = 30,
    batch_size: int = 64,
    config: TFTConfig | None = None,
    fast: bool = False,
) -> tuple:
    """Train TFT model and return (model, training_dataset, trainer).

    Args:
        feature_df: Combined feature DataFrame with time_idx and symbol columns
        max_prediction_length: Max forecast horizon
        max_epochs: Training epochs (reduced in --fast mode)
        batch_size: Batch size for DataLoader
        config: Model configuration
        fast: If True, use minimal epochs for quick testing

    Returns:
        (trained_model, training_dataset, trainer)
    """
    if fast:
        max_epochs = 2
        if config is None:
            config = TFTConfig(hidden_size=16, attention_head_size=1)

    training, validation = build_time_series_dataset(
        feature_df, max_prediction_length=max_prediction_length
    )

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, num_workers=0)

    model = create_tft_model(training, config)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = type(model).load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return best_model, training, trainer
```

**Step 5: Write predict.py**

```python
# backend/pipeline/predict.py
"""Generate predictions from trained TFT model."""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from backend.ml.model import QUANTILES


def generate_predictions(
    model,
    dataset,
    historical_df: pd.DataFrame,
    symbol: str,
    horizon: int = 5,
) -> dict:
    """Generate predictions for a single ticker at a specific horizon.

    Returns dict matching the API contract.
    """
    # Filter dataset for this symbol
    predict_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Get raw predictions (quantiles)
    raw_predictions = model.predict(predict_loader, mode="quantiles")

    # Extract quantile columns: [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    # Map to: lower_95, lower_80, _, median, _, upper_80, upper_95
    preds = raw_predictions.numpy() if hasattr(raw_predictions, "numpy") else np.array(raw_predictions)

    # Get the last prediction window
    if preds.ndim == 3:
        last_pred = preds[-1]  # shape: (horizon, n_quantiles)
    else:
        last_pred = preds

    # Build historical OHLCV (last 60 days)
    sym_df = historical_df[historical_df["symbol"] == symbol] if "symbol" in historical_df.columns else historical_df
    recent = sym_df.tail(60)

    historical = []
    for _, row in recent.iterrows():
        date_str = row.get("date", row.name)
        if hasattr(date_str, "strftime"):
            date_str = date_str.strftime("%Y-%m-%d")
        historical.append({
            "date": str(date_str),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    # Build prediction rows (up to requested horizon)
    last_date = pd.Timestamp(historical[-1]["date"])
    prediction_rows = []
    for i in range(min(horizon, last_pred.shape[0])):
        pred_date = last_date + pd.offsets.BDay(i + 1)
        q = last_pred[i]
        prediction_rows.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "median": round(float(q[3]), 2),      # 0.5 quantile
            "lower_80": round(float(q[1]), 2),     # 0.1 quantile
            "upper_80": round(float(q[5]), 2),     # 0.9 quantile
            "lower_95": round(float(q[0]), 2),     # 0.025 quantile
            "upper_95": round(float(q[6]), 2),     # 0.975 quantile
        })

    from datetime import datetime, timezone
    return {
        "symbol": symbol,
        "horizon_days": horizon,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "tft-v1",
        "historical": historical,
        "predictions": prediction_rows,
    }
```

**Step 6: Write export.py**

```python
# backend/pipeline/export.py
"""Export predictions and explainability data to JSON files."""
import json
import os
import tempfile
import shutil


def _atomic_write(data: dict, filepath: str) -> None:
    """Write JSON atomically: write to temp, then rename."""
    dir_path = os.path.dirname(filepath)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def export_predictions(predictions: dict, output_dir: str = "backend/predictions") -> str:
    """Export prediction dict to JSON file. Returns file path."""
    symbol = predictions["symbol"]
    horizon = predictions["horizon_days"]
    filename = f"{symbol}_h{horizon}.json"
    filepath = os.path.join(output_dir, filename)
    _atomic_write(predictions, filepath)
    return filepath


def export_explainability(explain_data: dict, output_dir: str = "backend/predictions") -> str:
    """Export explainability dict to JSON file. Returns file path."""
    symbol = explain_data["symbol"]
    horizon = explain_data["horizon_days"]
    filename = f"{symbol}_explain_h{horizon}.json"
    filepath = os.path.join(output_dir, filename)
    _atomic_write(explain_data, filepath)
    return filepath


def export_metadata(output_dir: str = "backend/predictions", model_version: str = "tft-v1") -> str:
    """Export metadata.json with run info."""
    from datetime import datetime, timezone
    from backend.pipeline.fetch_data import TICKER_UNIVERSE

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "tickers": [t["symbol"] for t in TICKER_UNIVERSE],
        "horizons": [1, 5, 10, 20],
    }
    filepath = os.path.join(output_dir, "metadata.json")
    _atomic_write(metadata, filepath)
    return filepath
```

**Step 7: Write run_pipeline.py**

```python
# backend/pipeline/run_pipeline.py
"""Pipeline orchestrator: fetch -> build -> train -> predict -> export."""
import argparse
import logging
from backend.pipeline.fetch_data import fetch_ohlcv, TICKER_UNIVERSE
from backend.pipeline.build_features import build_all_features
from backend.pipeline.train import train_tft
from backend.pipeline.predict import generate_predictions
from backend.pipeline.export import export_predictions, export_explainability, export_metadata
from backend.ml.explain import extract_feature_importance, extract_temporal_attention

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [1, 5, 10, 20]


def run(tickers: list[str] | None = None, fast: bool = False, output_dir: str = "backend/predictions"):
    """Run the full pipeline."""
    # 1. Determine ticker list
    if tickers:
        universe = [t for t in TICKER_UNIVERSE if t["symbol"] in tickers]
    else:
        universe = TICKER_UNIVERSE

    symbols = [t["symbol"] for t in universe]
    logger.info(f"Pipeline starting for {len(symbols)} tickers (fast={fast})")

    # 2. Fetch data
    logger.info("Fetching OHLCV data...")
    ohlcv_data = {}
    for sym in symbols:
        df = fetch_ohlcv(sym, period="2y")
        if not df.empty:
            ohlcv_data[sym] = df
            logger.info(f"  {sym}: {len(df)} rows")
        else:
            logger.warning(f"  {sym}: no data, skipping")

    # 3. Build features
    logger.info("Building features...")
    feature_df = build_all_features(ohlcv_data)
    logger.info(f"Feature DataFrame: {feature_df.shape}")

    # 4. Train model
    logger.info("Training TFT model...")
    model, training_dataset, trainer = train_tft(
        feature_df,
        max_prediction_length=max(HORIZONS),
        fast=fast,
    )
    logger.info("Training complete.")

    # 5. Generate predictions and explainability for each ticker/horizon
    logger.info("Generating predictions...")
    for sym in ohlcv_data.keys():
        for horizon in HORIZONS:
            try:
                pred = generate_predictions(
                    model, training_dataset, feature_df, sym, horizon=horizon
                )
                export_predictions(pred, output_dir=output_dir)
                logger.info(f"  {sym} h{horizon}: exported")
            except Exception as e:
                logger.error(f"  {sym} h{horizon}: failed — {e}")

    # 6. Export metadata
    export_metadata(output_dir=output_dir)
    logger.info(f"Pipeline complete. Output: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Astraeus prediction pipeline")
    parser.add_argument("--tickers", type=str, help="Comma-separated ticker list (default: all)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 2 epochs, small model")
    parser.add_argument("--output-dir", default="backend/predictions", help="Output directory")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    run(tickers=tickers, fast=args.fast, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

**Step 8: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_pipeline.py -v`
Expected: 2 tests pass

**Step 9: Commit**

```bash
git add backend/pipeline/ backend/tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator (fetch, build, train, predict, export)"
```

---

## Phase 1B: API Layer

### Task 7: Pydantic Schemas & Prediction Store Loader

**Files:**
- Create: `backend/api/schemas.py`
- Create: `backend/api/dependencies.py`

**Step 1: Write schemas.py**

```python
# backend/api/schemas.py
"""Pydantic response models matching API contract."""
from pydantic import BaseModel


class TickerInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    confidence_level: str  # "high" or "low"


class TickerListResponse(BaseModel):
    tickers: list[TickerInfo]


class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PredictionPoint(BaseModel):
    date: str
    median: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class PredictionResponse(BaseModel):
    symbol: str
    horizon_days: int
    generated_at: str
    model_version: str
    historical: list[OHLCVPoint]
    predictions: list[PredictionPoint]


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class TemporalAttention(BaseModel):
    date: str
    weight: float


class ExplainabilityResponse(BaseModel):
    symbol: str
    horizon_days: int
    feature_importance: list[FeatureImportance]
    temporal_attention: list[TemporalAttention]


class ErrorResponse(BaseModel):
    error: str
    detail: str
```

**Step 2: Write dependencies.py**

```python
# backend/api/dependencies.py
"""Prediction store loader — reads pre-computed JSON files."""
import json
import os

PREDICTIONS_DIR = os.environ.get("PREDICTIONS_DIR", "backend/predictions")
SAMPLE_DIR = os.path.join(os.path.dirname(PREDICTIONS_DIR), "predictions", "sample")


def get_predictions_dir() -> str:
    """Return the predictions directory, falling back to sample/ if empty."""
    if os.path.exists(PREDICTIONS_DIR) and any(
        f.endswith(".json") and f != "metadata.json"
        for f in os.listdir(PREDICTIONS_DIR)
    ):
        return PREDICTIONS_DIR
    if os.path.exists(SAMPLE_DIR):
        return SAMPLE_DIR
    return PREDICTIONS_DIR


def load_prediction(symbol: str, horizon: int) -> dict | None:
    """Load prediction JSON for a symbol/horizon. Returns None if not found."""
    pred_dir = get_predictions_dir()
    filepath = os.path.join(pred_dir, f"{symbol}_h{horizon}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def load_explainability(symbol: str, horizon: int) -> dict | None:
    """Load explainability JSON for a symbol/horizon. Returns None if not found."""
    pred_dir = get_predictions_dir()
    filepath = os.path.join(pred_dir, f"{symbol}_explain_h{horizon}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def load_metadata() -> dict | None:
    """Load metadata.json."""
    pred_dir = get_predictions_dir()
    filepath = os.path.join(pred_dir, "metadata.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)
```

**Step 3: Commit**

```bash
git add backend/api/schemas.py backend/api/dependencies.py
git commit -m "feat: add Pydantic schemas and prediction store loader"
```

---

### Task 8: FastAPI Routes

**Files:**
- Create: `backend/api/main.py`
- Create: `backend/api/routes/tickers.py`
- Create: `backend/api/routes/predictions.py`
- Create: `backend/api/routes/explainability.py`
- Create: `backend/tests/test_api.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_api.py
import pytest
import json
import os
import tempfile
from fastapi.testclient import TestClient


@pytest.fixture
def sample_dir():
    """Create a temp dir with sample prediction files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pred = {
            "symbol": "AAPL",
            "horizon_days": 5,
            "generated_at": "2026-02-25T00:00:00Z",
            "model_version": "tft-v1",
            "historical": [{"date": "2026-02-20", "open": 182.1, "high": 184.5, "low": 181.2, "close": 183.7, "volume": 54000000}],
            "predictions": [{"date": "2026-02-26", "median": 185.2, "lower_80": 183.1, "upper_80": 187.3, "lower_95": 181.5, "upper_95": 189.0}],
        }
        with open(os.path.join(tmpdir, "AAPL_h5.json"), "w") as f:
            json.dump(pred, f)

        explain = {
            "symbol": "AAPL",
            "horizon_days": 5,
            "feature_importance": [{"feature": "RSI_14", "importance": 0.3}],
            "temporal_attention": [{"date": "2026-02-20", "weight": 0.5}],
        }
        with open(os.path.join(tmpdir, "AAPL_explain_h5.json"), "w") as f:
            json.dump(explain, f)

        yield tmpdir


@pytest.fixture
def client(sample_dir):
    os.environ["PREDICTIONS_DIR"] = sample_dir
    from backend.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_tickers(client):
    r = client.get("/tickers")
    assert r.status_code == 200
    tickers = r.json()["tickers"]
    assert len(tickers) >= 50
    symbols = [t["symbol"] for t in tickers]
    assert "AAPL" in symbols


def test_get_tickers_search(client):
    r = client.get("/tickers?q=apple")
    assert r.status_code == 200
    tickers = r.json()["tickers"]
    assert any(t["symbol"] == "AAPL" for t in tickers)


def test_get_predictions(client):
    r = client.get("/predictions/AAPL?horizon=5")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    assert data["horizon_days"] == 5
    assert len(data["predictions"]) > 0


def test_get_predictions_not_found(client):
    r = client.get("/predictions/ZZZZ?horizon=5")
    assert r.status_code == 404
    assert r.json()["error"] == "ticker_not_found"


def test_get_predictions_invalid_horizon(client):
    r = client.get("/predictions/AAPL?horizon=3")
    assert r.status_code == 422


def test_get_explainability(client):
    r = client.get("/explainability/AAPL?horizon=5")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    assert len(data["feature_importance"]) > 0


def test_get_explainability_not_found(client):
    r = client.get("/explainability/ZZZZ?horizon=5")
    assert r.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_api.py -v`
Expected: FAIL — import errors

**Step 3: Write main.py**

```python
# backend/api/main.py
"""FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import tickers, predictions, explainability

app = FastAPI(
    title="Astraeus",
    description="Research-grade financial forecasting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickers.router)
app.include_router(predictions.router)
app.include_router(explainability.router)


@app.get("/health")
def health():
    return {"status": "ok"}
```

**Step 4: Write route files**

```python
# backend/api/routes/tickers.py
from fastapi import APIRouter, Query
from backend.api.schemas import TickerListResponse, TickerInfo
from backend.pipeline.fetch_data import TICKER_UNIVERSE

router = APIRouter()


@router.get("/tickers", response_model=TickerListResponse)
def list_tickers(q: str = Query(default=None, description="Search query")):
    tickers = []
    for t in TICKER_UNIVERSE:
        info = TickerInfo(
            symbol=t["symbol"],
            name=t["name"],
            sector=t["sector"],
            confidence_level="high",  # Cold start detection done at pipeline time
        )
        if q:
            query = q.lower()
            if query in t["symbol"].lower() or query in t["name"].lower():
                tickers.append(info)
        else:
            tickers.append(info)
    return TickerListResponse(tickers=tickers)
```

```python
# backend/api/routes/predictions.py
from fastapi import APIRouter, Path, Query, HTTPException
from backend.api.schemas import PredictionResponse, ErrorResponse
from backend.api.dependencies import load_prediction

router = APIRouter()

VALID_HORIZONS = {1, 5, 10, 20}


@router.get(
    "/predictions/{symbol}",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def get_predictions(
    symbol: str = Path(description="Ticker symbol"),
    horizon: int = Query(default=5, description="Forecast horizon in trading days"),
):
    if horizon not in VALID_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_horizon", "detail": f"Horizon must be one of {sorted(VALID_HORIZONS)}"},
        )

    data = load_prediction(symbol.upper(), horizon)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "ticker_not_found", "detail": f"No predictions for {symbol.upper()}"},
        )

    return data
```

```python
# backend/api/routes/explainability.py
from fastapi import APIRouter, Path, Query, HTTPException
from backend.api.schemas import ExplainabilityResponse, ErrorResponse
from backend.api.dependencies import load_explainability

router = APIRouter()

VALID_HORIZONS = {1, 5, 10, 20}


@router.get(
    "/explainability/{symbol}",
    response_model=ExplainabilityResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def get_explainability(
    symbol: str = Path(description="Ticker symbol"),
    horizon: int = Query(default=5, description="Forecast horizon in trading days"),
):
    if horizon not in VALID_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_horizon", "detail": f"Horizon must be one of {sorted(VALID_HORIZONS)}"},
        )

    data = load_explainability(symbol.upper(), horizon)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "ticker_not_found", "detail": f"No explainability data for {symbol.upper()}"},
        )

    return data
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_api.py -v`
Expected: 8 tests pass

**Step 6: Commit**

```bash
git add backend/api/ backend/tests/test_api.py
git commit -m "feat: add FastAPI routes (tickers, predictions, explainability)"
```

---

## Phase 1C: Sample Data

### Task 9: Generate Sample Predictions for Frontend Development

**Files:**
- Create: `backend/predictions/sample/AAPL_h1.json`
- Create: `backend/predictions/sample/AAPL_h5.json`
- Create: `backend/predictions/sample/AAPL_h10.json`
- Create: `backend/predictions/sample/AAPL_h20.json`
- Create: `backend/predictions/sample/AAPL_explain_h5.json`
- Create: `backend/predictions/sample/metadata.json`
- Create: `scripts/generate_sample_data.py`

**Step 1: Write sample data generator**

```python
# scripts/generate_sample_data.py
"""Generate realistic sample prediction data for frontend development."""
import json
import os
import numpy as np
from datetime import datetime, timezone

np.random.seed(42)
OUTPUT_DIR = "backend/predictions/sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_TICKERS = [
    {"symbol": "AAPL", "base_price": 185.0},
    {"symbol": "NVDA", "base_price": 870.0},
    {"symbol": "MSFT", "base_price": 410.0},
    {"symbol": "XOM", "base_price": 105.0},
    {"symbol": "CVX", "base_price": 155.0},
]

HORIZONS = [1, 5, 10, 20]

FEATURES = [
    "RSI_14", "MACD_12_26_9", "Close_lag_5", "Volume", "ATR_14",
    "fed_funds_rate", "EMA_20", "BBU_20_2.0", "return_lag_1", "OBV",
]


def generate_historical(base_price: float, n_days: int = 60) -> list[dict]:
    prices = [base_price]
    for _ in range(n_days - 1):
        change = np.random.randn() * base_price * 0.015
        prices.append(prices[-1] + change)

    rows = []
    from pandas import bdate_range
    dates = bdate_range(end="2026-02-24", periods=n_days)
    for i, date in enumerate(dates):
        p = prices[i]
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(p + np.random.randn() * 0.5, 2),
            "high": round(p + abs(np.random.randn()) * 2, 2),
            "low": round(p - abs(np.random.randn()) * 2, 2),
            "close": round(p, 2),
            "volume": int(np.random.randint(20_000_000, 80_000_000)),
        })
    return rows


def generate_predictions(base_price: float, horizon: int) -> list[dict]:
    from pandas import bdate_range
    dates = bdate_range(start="2026-02-25", periods=horizon)
    rows = []
    p = base_price
    for i, date in enumerate(dates):
        drift = np.random.randn() * base_price * 0.01
        p += drift
        spread_80 = base_price * 0.02 * (i + 1) ** 0.5
        spread_95 = base_price * 0.035 * (i + 1) ** 0.5
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "median": round(p, 2),
            "lower_80": round(p - spread_80, 2),
            "upper_80": round(p + spread_80, 2),
            "lower_95": round(p - spread_95, 2),
            "upper_95": round(p + spread_95, 2),
        })
    return rows


def generate_explainability(symbol: str, horizon: int) -> dict:
    importances = np.random.dirichlet(np.ones(len(FEATURES)))
    importances.sort()
    importances = importances[::-1]

    from pandas import bdate_range
    dates = bdate_range(end="2026-02-24", periods=60)
    attention = np.random.dirichlet(np.ones(60))

    return {
        "symbol": symbol,
        "horizon_days": horizon,
        "feature_importance": [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in zip(FEATURES, importances)
        ],
        "temporal_attention": [
            {"date": d.strftime("%Y-%m-%d"), "weight": round(float(w), 6)}
            for d, w in zip(dates, attention)
        ],
    }


if __name__ == "__main__":
    for ticker in SAMPLE_TICKERS:
        sym = ticker["symbol"]
        base = ticker["base_price"]
        hist = generate_historical(base)

        for h in HORIZONS:
            pred_data = {
                "symbol": sym,
                "horizon_days": h,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_version": "tft-v1-sample",
                "historical": hist,
                "predictions": generate_predictions(base, h),
            }
            filepath = os.path.join(OUTPUT_DIR, f"{sym}_h{h}.json")
            with open(filepath, "w") as f:
                json.dump(pred_data, f, indent=2)
            print(f"  Wrote {filepath}")

        # Explainability for all horizons
        for h in HORIZONS:
            explain = generate_explainability(sym, h)
            filepath = os.path.join(OUTPUT_DIR, f"{sym}_explain_h{h}.json")
            with open(filepath, "w") as f:
                json.dump(explain, f, indent=2)
            print(f"  Wrote {filepath}")

    # Metadata
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "tft-v1-sample",
        "tickers": [t["symbol"] for t in SAMPLE_TICKERS],
        "horizons": HORIZONS,
        "note": "Sample data for frontend development. Not real predictions.",
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. Sample data generated.")
```

**Step 2: Run the generator**

Run: `python scripts/generate_sample_data.py`
Expected: Files created in `backend/predictions/sample/`

**Step 3: Verify API serves sample data**

Run: `cd C:/Users/King\ Hratch/astraeus && python -m uvicorn backend.api.main:app --port 8000 &`
Then: `curl http://localhost:8000/predictions/AAPL?horizon=5`
Expected: JSON response with sample predictions

**Step 4: Commit**

```bash
git add scripts/generate_sample_data.py backend/predictions/sample/
git commit -m "feat: add sample prediction data for frontend development"
```

---

## Phase 1D: Frontend

### Task 10: Next.js Scaffolding + Shadcn + Dark Theme

**Step 1: Create Next.js project**

Run:
```bash
cd C:/Users/King\ Hratch/astraeus
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --use-npm
```

**Step 2: Install dependencies**

Run:
```bash
cd C:/Users/King\ Hratch/astraeus/frontend
npx shadcn@latest init -d
npm install lightweight-charts recharts
```

**Step 3: Configure dark theme in globals.css**

Update `frontend/app/globals.css` to include fintech dark theme:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 240 10% 3.9%;
    --foreground: 0 0% 95%;
    --card: 240 10% 5.9%;
    --card-foreground: 0 0% 95%;
    --primary: 142.1 76.2% 36.3%;
    --primary-foreground: 355.7 100% 97.3%;
    --muted: 240 3.7% 15.9%;
    --muted-foreground: 240 5% 64.9%;
    --destructive: 0 62.8% 50.6%;
    --border: 240 3.7% 15.9%;
    --ring: 142.1 76.2% 36.3%;
    --radius: 0.5rem;
  }
}

@layer base {
  body {
    @apply bg-background text-foreground;
    font-feature-settings: "tnum";
  }
}
```

**Step 4: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold Next.js frontend with Shadcn UI and dark theme"
```

---

### Task 11: TypeScript Types & API Client

**Files:**
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/api.ts`

**Step 1: Write types.ts**

```typescript
// frontend/lib/types.ts
export interface TickerInfo {
  symbol: string;
  name: string;
  sector: string;
  confidence_level: "high" | "low";
}

export interface OHLCVPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionPoint {
  date: string;
  median: number;
  lower_80: number;
  upper_80: number;
  lower_95: number;
  upper_95: number;
}

export interface PredictionResponse {
  symbol: string;
  horizon_days: number;
  generated_at: string;
  model_version: string;
  historical: OHLCVPoint[];
  predictions: PredictionPoint[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface TemporalAttention {
  date: string;
  weight: number;
}

export interface ExplainabilityResponse {
  symbol: string;
  horizon_days: number;
  feature_importance: FeatureImportance[];
  temporal_attention: TemporalAttention[];
}

export type Horizon = 1 | 5 | 10 | 20;
```

**Step 2: Write api.ts**

```typescript
// frontend/lib/api.ts
import type {
  TickerInfo,
  PredictionResponse,
  ExplainabilityResponse,
  Horizon,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "unknown", detail: res.statusText }));
    throw new Error(error.detail || error.error || "API error");
  }
  return res.json();
}

export async function getTickers(query?: string): Promise<TickerInfo[]> {
  const params = query ? `?q=${encodeURIComponent(query)}` : "";
  const data = await fetchJSON<{ tickers: TickerInfo[] }>(`/tickers${params}`);
  return data.tickers;
}

export async function getPredictions(symbol: string, horizon: Horizon): Promise<PredictionResponse> {
  return fetchJSON<PredictionResponse>(`/predictions/${symbol}?horizon=${horizon}`);
}

export async function getExplainability(symbol: string, horizon: Horizon): Promise<ExplainabilityResponse> {
  return fetchJSON<ExplainabilityResponse>(`/explainability/${symbol}?horizon=${horizon}`);
}
```

**Step 3: Commit**

```bash
git add frontend/lib/
git commit -m "feat: add TypeScript types and API client"
```

---

### Task 12: Ticker Search Sidebar

**Files:**
- Create: `frontend/components/TickerSearch.tsx`
- Modify: `frontend/app/layout.tsx`
- Modify: `frontend/app/page.tsx`

**Step 1: Install Shadcn input component**

Run:
```bash
cd C:/Users/King\ Hratch/astraeus/frontend
npx shadcn@latest add input badge scroll-area
```

**Step 2: Write TickerSearch.tsx**

```tsx
// frontend/components/TickerSearch.tsx
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getTickers } from "@/lib/api";
import type { TickerInfo } from "@/lib/types";

export function TickerSearch() {
  const [query, setQuery] = useState("");
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const timeout = setTimeout(() => {
      setLoading(true);
      getTickers(query || undefined)
        .then(setTickers)
        .catch(() => setTickers([]))
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(timeout);
  }, [query]);

  const sectors = [...new Set(tickers.map((t) => t.sector))];

  return (
    <div className="w-64 border-r border-border flex flex-col h-full">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-bold tracking-tight mb-3">Astraeus</h1>
        <Input
          placeholder="Search tickers..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="bg-muted"
        />
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2">
          {sectors.map((sector) => (
            <div key={sector} className="mb-4">
              <p className="text-xs font-medium text-muted-foreground px-2 mb-1">
                {sector}
              </p>
              {tickers
                .filter((t) => t.sector === sector)
                .map((t) => (
                  <button
                    key={t.symbol}
                    onClick={() => router.push(`/ticker/${t.symbol}`)}
                    className="w-full text-left px-2 py-1.5 rounded-md hover:bg-muted flex items-center justify-between group"
                  >
                    <div>
                      <span className="font-mono text-sm font-medium">
                        {t.symbol}
                      </span>
                      <span className="text-xs text-muted-foreground ml-2">
                        {t.name}
                      </span>
                    </div>
                    {t.confidence_level === "low" && (
                      <Badge variant="outline" className="text-amber-500 border-amber-500 text-[10px]">
                        Low
                      </Badge>
                    )}
                  </button>
                ))}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
```

**Step 3: Update layout.tsx**

```tsx
// frontend/app/layout.tsx
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { TickerSearch } from "@/components/TickerSearch";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Astraeus — Financial Forecasting Dashboard",
  description: "Research-grade stock market predictions powered by Temporal Fusion Transformers",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${mono.variable} font-sans antialiased`}>
        <div className="flex h-screen overflow-hidden">
          <TickerSearch />
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
```

**Step 4: Update page.tsx (landing)**

```tsx
// frontend/app/page.tsx
export default function Home() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <h2 className="text-2xl font-bold tracking-tight mb-2">Astraeus</h2>
        <p className="text-muted-foreground">
          Select a ticker from the sidebar to view predictions.
        </p>
      </div>
    </div>
  );
}
```

**Step 5: Verify**

Run: `cd C:/Users/King\ Hratch/astraeus/frontend && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: add ticker search sidebar with sector grouping"
```

---

### Task 13: Candlestick Chart with Lightweight Charts

**Files:**
- Create: `frontend/components/charts/CandlestickChart.tsx`
- Create: `frontend/components/charts/ConfidenceBands.tsx`
- Create: `frontend/components/HorizonToggle.tsx`
- Create: `frontend/app/ticker/[symbol]/page.tsx`

**Step 1: Write CandlestickChart.tsx**

```tsx
// frontend/components/charts/CandlestickChart.tsx
"use client";

import { useEffect, useRef } from "react";
import { createChart, CandlestickSeries, AreaSeries } from "lightweight-charts";
import type { OHLCVPoint, PredictionPoint } from "@/lib/types";

interface Props {
  historical: OHLCVPoint[];
  predictions: PredictionPoint[];
}

export function CandlestickChart({ historical, predictions }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#a1a1aa",
        fontFamily: "var(--font-mono)",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
      timeScale: { borderColor: "rgba(255,255,255,0.1)" },
    });

    // Candlestick series for historical data
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    candleSeries.setData(
      historical.map((p) => ({
        time: p.date,
        open: p.open,
        high: p.high,
        low: p.low,
        close: p.close,
      }))
    );

    // 95% confidence band (lighter)
    const band95 = chart.addSeries(AreaSeries, {
      lineColor: "rgba(34, 197, 94, 0.0)",
      topColor: "rgba(34, 197, 94, 0.08)",
      bottomColor: "rgba(34, 197, 94, 0.02)",
      lineWidth: 0,
    });

    // 80% confidence band (darker)
    const band80 = chart.addSeries(AreaSeries, {
      lineColor: "rgba(34, 197, 94, 0.0)",
      topColor: "rgba(34, 197, 94, 0.15)",
      bottomColor: "rgba(34, 197, 94, 0.05)",
      lineWidth: 0,
    });

    // Median prediction line
    const medianLine = chart.addSeries(AreaSeries, {
      lineColor: "#22c55e",
      topColor: "rgba(34, 197, 94, 0.0)",
      bottomColor: "rgba(34, 197, 94, 0.0)",
      lineWidth: 2,
      lineStyle: 2, // dashed
    });

    if (predictions.length > 0) {
      band95.setData(predictions.map((p) => ({ time: p.date, value: p.upper_95 })));
      band80.setData(predictions.map((p) => ({ time: p.date, value: p.upper_80 })));
      medianLine.setData(predictions.map((p) => ({ time: p.date, value: p.median })));
    }

    chart.timeScale().fitContent();

    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, [historical, predictions]);

  return <div ref={containerRef} className="w-full h-[500px]" />;
}
```

**Step 2: Write HorizonToggle.tsx**

```tsx
// frontend/components/HorizonToggle.tsx
"use client";

import type { Horizon } from "@/lib/types";

interface Props {
  value: Horizon;
  onChange: (h: Horizon) => void;
}

const HORIZONS: Horizon[] = [1, 5, 10, 20];
const LABELS: Record<Horizon, string> = { 1: "1D", 5: "5D", 10: "10D", 20: "20D" };

export function HorizonToggle({ value, onChange }: Props) {
  return (
    <div className="flex gap-1">
      {HORIZONS.map((h) => (
        <button
          key={h}
          onClick={() => onChange(h)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            value === h
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground"
          }`}
        >
          {LABELS[h]}
        </button>
      ))}
    </div>
  );
}
```

**Step 3: Write ticker detail page**

```tsx
// frontend/app/ticker/[symbol]/page.tsx
"use client";

import { useState, useEffect, use } from "react";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { HorizonToggle } from "@/components/HorizonToggle";
import { getPredictions, getExplainability } from "@/lib/api";
import type { PredictionResponse, ExplainabilityResponse, Horizon } from "@/lib/types";

type Tab = "chart" | "explainability";

export default function TickerPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = use(params);
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [tab, setTab] = useState<Tab>("chart");
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null);
  const [explainability, setExplainability] = useState<ExplainabilityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    getPredictions(symbol, horizon)
      .then(setPredictions)
      .catch((e) => setError(e.message));
    getExplainability(symbol, horizon)
      .then(setExplainability)
      .catch(() => {}); // Explainability is optional
  }, [symbol, horizon]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-destructive text-lg font-medium">Error</p>
          <p className="text-muted-foreground mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!predictions) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold font-mono">{symbol}</h2>
          <p className="text-xs text-muted-foreground">
            Model: {predictions.model_version} | Generated: {new Date(predictions.generated_at).toLocaleDateString()}
          </p>
        </div>
        <HorizonToggle value={horizon} onChange={setHorizon} />
      </div>

      {/* Tab toggle */}
      <div className="flex gap-1 mb-4">
        <button
          onClick={() => setTab("chart")}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            tab === "chart" ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Chart
        </button>
        <button
          onClick={() => setTab("explainability")}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            tab === "explainability" ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          Explainability
        </button>
      </div>

      {/* Content */}
      {tab === "chart" && (
        <div className="bg-card rounded-lg border border-border p-4">
          <CandlestickChart
            historical={predictions.historical}
            predictions={predictions.predictions}
          />
        </div>
      )}

      {tab === "explainability" && explainability && (
        <div className="text-muted-foreground">
          {/* Placeholder — implemented in Task 14 */}
          Explainability visualizations will render here.
        </div>
      )}
    </div>
  );
}
```

**Step 4: Verify**

Run: `cd C:/Users/King\ Hratch/astraeus/frontend && npm run build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: add candlestick chart, confidence bands, horizon toggle, and ticker page"
```

---

### Task 14: Explainability Tab (Feature Importance + Attention Heatmap)

**Files:**
- Create: `frontend/components/charts/FeatureImportanceChart.tsx`
- Create: `frontend/components/charts/AttentionHeatmap.tsx`
- Modify: `frontend/app/ticker/[symbol]/page.tsx` (replace placeholder)

**Step 1: Install Recharts**

Already installed in Task 10. Verify: `npm ls recharts`

**Step 2: Write FeatureImportanceChart.tsx**

```tsx
// frontend/components/charts/FeatureImportanceChart.tsx
"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { FeatureImportance } from "@/lib/types";

interface Props {
  data: FeatureImportance[];
}

export function FeatureImportanceChart({ data }: Props) {
  const top10 = data.slice(0, 10);

  return (
    <div>
      <h3 className="text-sm font-medium mb-3">Feature Importance</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={top10} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" domain={[0, "auto"]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "var(--font-mono)" }}
            width={75}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
            labelStyle={{ color: "#fafafa" }}
            itemStyle={{ color: "#22c55e" }}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {top10.map((_, i) => (
              <Cell key={i} fill={`rgba(34, 197, 94, ${1 - i * 0.08})`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

**Step 3: Write AttentionHeatmap.tsx**

```tsx
// frontend/components/charts/AttentionHeatmap.tsx
"use client";

import type { TemporalAttention } from "@/lib/types";

interface Props {
  data: TemporalAttention[];
}

export function AttentionHeatmap({ data }: Props) {
  const maxWeight = Math.max(...data.map((d) => d.weight));

  return (
    <div>
      <h3 className="text-sm font-medium mb-3">Temporal Attention</h3>
      <p className="text-xs text-muted-foreground mb-2">
        Darker cells = model paid more attention to that day
      </p>
      <div className="flex flex-wrap gap-[2px]">
        {data.map((d) => {
          const intensity = d.weight / maxWeight;
          return (
            <div
              key={d.date}
              title={`${d.date}: ${(d.weight * 100).toFixed(2)}%`}
              className="w-4 h-4 rounded-sm cursor-pointer transition-transform hover:scale-150"
              style={{
                backgroundColor: `rgba(34, 197, 94, ${0.1 + intensity * 0.9})`,
              }}
            />
          );
        })}
      </div>
      <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
        <span>Low</span>
        <div className="flex gap-[1px]">
          {[0.1, 0.3, 0.5, 0.7, 0.9].map((v) => (
            <div
              key={v}
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: `rgba(34, 197, 94, ${v})` }}
            />
          ))}
        </div>
        <span>High</span>
      </div>
    </div>
  );
}
```

**Step 4: Update ticker page to use real components**

Replace the explainability placeholder in `frontend/app/ticker/[symbol]/page.tsx`:

```tsx
// Replace the explainability section with:
{tab === "explainability" && explainability && (
  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <div className="bg-card rounded-lg border border-border p-4">
      <FeatureImportanceChart data={explainability.feature_importance} />
    </div>
    <div className="bg-card rounded-lg border border-border p-4">
      <AttentionHeatmap data={explainability.temporal_attention} />
    </div>
  </div>
)}
```

Add imports at the top:
```tsx
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { AttentionHeatmap } from "@/components/charts/AttentionHeatmap";
```

**Step 5: Verify**

Run: `cd C:/Users/King\ Hratch/astraeus/frontend && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: add explainability tab with feature importance chart and attention heatmap"
```

---

## Phase 1E: Integration

### Task 15: Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `backend/Dockerfile`
- Modify: `frontend/next.config.ts` (add API rewrite for dev)

**Step 1: Write backend Dockerfile**

```dockerfile
# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Write docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.8"

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend
    environment:
      - PREDICTIONS_DIR=/app/backend/predictions
      - FRED_API_KEY=${FRED_API_KEY:-}

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
```

**Step 3: Verify**

Run: `docker compose config`
Expected: Valid YAML output

**Step 4: Commit**

```bash
git add docker-compose.yml backend/Dockerfile
git commit -m "feat: add Docker Compose for local development"
```

---

### Task 16: README

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

```markdown
# Astraeus

Research-grade financial forecasting dashboard powered by Temporal Fusion Transformers.

Generates probabilistic multi-horizon stock predictions (1, 5, 10, 20 trading days) with model explainability visualizations. Built for a professional portfolio — not a tutorial project.

## Architecture

- **Backend:** Python + PyTorch Forecasting + FastAPI
- **Frontend:** Next.js 15 + Shadcn UI + Tailwind + Lightweight Charts
- **ML Model:** Temporal Fusion Transformer (TFT) with quantile regression
- **Deploy:** Vercel (frontend) + Railway (API) — hybrid pre-computed predictions

## Quick Start

### Prerequisites

- Python 3.10-3.12
- Node.js 20+
- Docker (optional)

### Backend

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r backend/requirements.txt

# Run with sample data
uvicorn backend.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

### Run the ML Pipeline

```bash
# Quick test (2 tickers, fast training)
python -m backend.pipeline.run_pipeline --tickers AAPL,NVDA --fast

# Full pipeline (all 50 tickers)
python -m backend.pipeline.run_pipeline
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /tickers?q=` | List/search tickers |
| `GET /predictions/{symbol}?horizon=5` | Predictions with confidence intervals |
| `GET /explainability/{symbol}?horizon=5` | Feature importance + temporal attention |

## Project Structure

```
astraeus/
├── backend/
│   ├── api/          # FastAPI serving layer
│   ├── ml/           # Model code (TFT, features, explainability)
│   ├── pipeline/     # Offline data → train → predict → export
│   ├── predictions/  # Pre-computed prediction store
│   └── tests/
├── frontend/
│   ├── app/          # Next.js App Router
│   ├── components/   # Shadcn + chart components
│   └── lib/          # API client, types
└── docs/plans/       # Design and implementation docs
```

## Security

- No API keys or credentials are committed to this repository
- All secrets are managed via `.env` (see `.env.example`)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README"
```

---

## Summary

| Phase | Tasks | Description |
|---|---|---|
| 1A | 1-6 | Backend foundation: scaffolding, data fetch, features, TFT model, explainability, pipeline |
| 1B | 7-8 | API layer: schemas, routes (tickers, predictions, explainability) |
| 1C | 9 | Sample data for frontend development |
| 1D | 10-14 | Frontend: Next.js scaffold, types, ticker search, candlestick chart, explainability tab |
| 1E | 15-16 | Integration: Docker Compose, README |

**Total: 16 tasks, TDD throughout, frequent commits.**

After Phase 1 is complete, Phase 2 adds: GNN relational layer, backtesting view, daily auto-refresh, macro indicators panel.
