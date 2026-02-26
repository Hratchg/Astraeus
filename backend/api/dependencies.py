"""Prediction store loader — reads pre-computed JSON files."""
import json
import os

def get_predictions_dir() -> str:
    """Return the predictions directory, falling back to sample/ if empty."""
    pred_dir = os.environ.get("PREDICTIONS_DIR", "backend/predictions")
    sample_dir = os.path.join(os.path.dirname(pred_dir), "predictions", "sample")

    if os.path.exists(pred_dir) and any(
        f.endswith(".json") and f != "metadata.json"
        for f in os.listdir(pred_dir)
    ):
        return pred_dir
    if os.path.exists(sample_dir):
        return sample_dir
    return pred_dir


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
