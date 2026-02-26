"""Generate realistic sample prediction data for frontend development."""
import json
import os
import numpy as np
from datetime import datetime, timezone
from pandas import bdate_range

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "predictions", "sample")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)
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
    dates = bdate_range(end="2026-02-24", periods=n_days)
    for i, date in enumerate(dates):
        p = prices[i]
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(p + np.random.randn() * 0.5, 2),
            "high": round(p + abs(np.random.randn()) * 2, 2),
            "low": round(p - abs(np.random.randn()) * 2, 2),
            "close": round(float(p), 2),
            "volume": int(np.random.randint(20_000_000, 80_000_000)),
        })
    return rows


def generate_predictions(base_price: float, horizon: int) -> list[dict]:
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
            "median": round(float(p), 2),
            "lower_80": round(float(p - spread_80), 2),
            "upper_80": round(float(p + spread_80), 2),
            "lower_95": round(float(p - spread_95), 2),
            "upper_95": round(float(p + spread_95), 2),
        })
    return rows


def generate_explainability(symbol: str, horizon: int) -> dict:
    importances = np.random.dirichlet(np.ones(len(FEATURES)))
    importances.sort()
    importances = importances[::-1]

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
