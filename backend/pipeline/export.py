"""Export pipeline results to JSON files.

Provides atomic writes and structured export functions for predictions,
explainability data, and pipeline metadata.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone


def _atomic_write(data: dict, filepath: str) -> None:
    """Write JSON data atomically using a temp file + rename.

    Parameters
    ----------
    data : dict
        JSON-serialisable dictionary to write.
    filepath : str
        Destination file path.
    """
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)

    # Write to a temp file in the same directory, then rename.
    # This avoids partial writes if the process is interrupted.
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        # On Windows, os.rename fails if the target exists, so remove first.
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmp_path, filepath)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def export_predictions(predictions: dict, output_dir: str) -> str:
    """Write prediction data to ``{SYMBOL}_h{horizon}.json``.

    Parameters
    ----------
    predictions : dict
        Must contain ``symbol`` and ``horizon_days`` keys.
    output_dir : str
        Directory to write the file to.

    Returns
    -------
    str
        Full path of the written file.
    """
    symbol = predictions["symbol"]
    horizon = predictions["horizon_days"]
    filename = f"{symbol}_h{horizon}.json"
    filepath = os.path.join(output_dir, filename)
    _atomic_write(predictions, filepath)
    return filepath


def export_explainability(explain_data: dict, output_dir: str) -> str:
    """Write explainability data to ``{SYMBOL}_explain_h{horizon}.json``.

    Parameters
    ----------
    explain_data : dict
        Must contain ``symbol`` and ``horizon_days`` keys.
    output_dir : str
        Directory to write the file to.

    Returns
    -------
    str
        Full path of the written file.
    """
    symbol = explain_data["symbol"]
    horizon = explain_data["horizon_days"]
    filename = f"{symbol}_explain_h{horizon}.json"
    filepath = os.path.join(output_dir, filename)
    _atomic_write(explain_data, filepath)
    return filepath


def export_metadata(
    output_dir: str,
    model_version: str = "tft-v1",
    tickers: list[str] | None = None,
    horizons: list[int] | None = None,
) -> str:
    """Write pipeline metadata to ``metadata.json``.

    Parameters
    ----------
    output_dir : str
        Directory to write the file to.
    model_version : str
        Model version string.
    tickers : list[str], optional
        List of ticker symbols processed.
    horizons : list[int], optional
        List of forecast horizons generated.

    Returns
    -------
    str
        Full path of the written file.
    """
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "tickers": tickers or [],
        "horizons": horizons or [],
    }
    filepath = os.path.join(output_dir, "metadata.json")
    _atomic_write(metadata, filepath)
    return filepath
