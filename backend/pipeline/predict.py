"""Generate predictions from trained TFT model."""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from backend.ml.model import QUANTILES


def generate_predictions(model, dataset, historical_df, symbol, horizon=5):
    """Generate predictions for a single ticker at a specific horizon.
    Returns dict matching the API contract."""
    predict_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    raw_predictions = model.predict(predict_loader, mode="quantiles")

    preds = raw_predictions.numpy() if hasattr(raw_predictions, "numpy") else np.array(raw_predictions)

    if preds.ndim == 3:
        last_pred = preds[-1]
    else:
        last_pred = preds

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

    last_date = pd.Timestamp(historical[-1]["date"])
    prediction_rows = []
    for i in range(min(horizon, last_pred.shape[0])):
        pred_date = last_date + pd.offsets.BDay(i + 1)
        q = last_pred[i]
        prediction_rows.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "median": round(float(q[3]), 2),
            "lower_80": round(float(q[1]), 2),
            "upper_80": round(float(q[5]), 2),
            "lower_95": round(float(q[0]), 2),
            "upper_95": round(float(q[6]), 2),
        })

    return {
        "symbol": symbol,
        "horizon_days": horizon,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "tft-v1",
        "historical": historical,
        "predictions": prediction_rows,
    }
