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
