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
