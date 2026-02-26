"""Prediction store loader — reads pre-computed JSON files."""
import json
import logging
import os

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PRED_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "predictions"))
_SAMPLE_DIR = os.path.join(_DEFAULT_PRED_DIR, "sample")


def get_predictions_dir() -> str:
    """Return the predictions directory, falling back to sample/ if empty."""
    pred_dir = os.environ.get("PREDICTIONS_DIR", _DEFAULT_PRED_DIR)

    if os.path.exists(pred_dir) and any(
        f.endswith(".json") and f != "metadata.json"
        for f in os.listdir(pred_dir)
    ):
        return pred_dir
    if os.path.exists(_SAMPLE_DIR):
        return _SAMPLE_DIR
    return pred_dir


def _load_json(filepath: str) -> dict | None:
    """Load a JSON file, returning None on missing or corrupt files."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Corrupt JSON file: %s", filepath)
        return None


def load_prediction(symbol: str, horizon: int) -> dict | None:
    """Load prediction JSON for a symbol/horizon. Returns None if not found."""
    pred_dir = get_predictions_dir()
    return _load_json(os.path.join(pred_dir, f"{symbol}_h{horizon}.json"))


def load_explainability(symbol: str, horizon: int) -> dict | None:
    """Load explainability JSON for a symbol/horizon. Returns None if not found."""
    pred_dir = get_predictions_dir()
    return _load_json(os.path.join(pred_dir, f"{symbol}_explain_h{horizon}.json"))
