"""
Explainability extraction for Astraeus.

Utility functions to extract human-readable explainability data from
TFT model outputs (feature importance and temporal attention weights).
These are called by the pipeline to generate JSON served by the API
to the frontend's explainability tab.

Only uses numpy — no PyTorch dependency needed in this module.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def extract_feature_importance(interpretation: dict) -> list[dict]:
    """Extract and rank feature importances from a TFT interpretation dict.

    Parameters
    ----------
    interpretation : dict
        Must contain:
        - ``encoder_variables``: list of feature name strings
        - ``encoder_importance``: list of floats or tensors

    Returns
    -------
    list[dict]
        List of ``{"feature": str, "importance": float}`` dicts,
        sorted descending by importance.  Importances are rounded
        to 4 decimal places.
    """
    variables: list[str] = interpretation["encoder_variables"]
    importances = interpretation["encoder_importance"]

    # Convert tensors to plain floats if necessary
    plain_importances: list[float] = []
    for val in importances:
        if hasattr(val, "item"):
            # PyTorch tensor or numpy scalar — call .item()
            plain_importances.append(float(val.item()))
        else:
            plain_importances.append(float(val))

    result = [
        {"feature": var, "importance": round(imp, 4)}
        for var, imp in zip(variables, plain_importances)
    ]

    # Sort descending by importance
    result.sort(key=lambda x: x["importance"], reverse=True)

    return result


# ---------------------------------------------------------------------------
# Temporal attention
# ---------------------------------------------------------------------------


def extract_temporal_attention(
    attention_weights: np.ndarray,
    dates: list[str],
) -> list[dict]:
    """Extract normalised temporal attention weights paired with dates.

    Parameters
    ----------
    attention_weights : np.ndarray
        1-D array of raw attention weights (one per timestep).
    dates : list[str]
        Date strings corresponding to each timestep.

    Returns
    -------
    list[dict]
        List of ``{"date": str, "weight": float}`` dicts.
        Weights are normalised to sum to 1 and rounded to 6 decimal places.
    """
    weights = np.asarray(attention_weights, dtype=np.float64)

    # Normalise so weights sum to 1
    total = weights.sum()
    if total != 0:
        weights = weights / total

    result = [
        {"date": date, "weight": round(float(w), 6)}
        for date, w in zip(dates, weights)
    ]

    return result
