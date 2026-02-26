"""
TFT dataset builder for Astraeus.

Wraps pytorch_forecasting.TimeSeriesDataSet to produce training and
validation splits from a feature-engineered DataFrame.
"""

from __future__ import annotations

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_VARYING_UNKNOWN: list[str] = [
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "RSI_14",
    "MACD_12_26_9",
    "ATR_14",
    "return_lag_1",
    "return_lag_5",
]
"""Feature columns the TFT model uses as time-varying unknown reals."""

MAX_ENCODER_LENGTH: int = 60
"""Look-back window: 60 trading days (~3 months)."""


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_time_series_dataset(
    df: pd.DataFrame,
    max_prediction_length: int = 20,
    training_cutoff_frac: float = 0.8,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build training and validation TimeSeriesDataSets from a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``symbol``, ``time_idx``, and the feature
        columns listed in :data:`TIME_VARYING_UNKNOWN`.
    max_prediction_length : int
        Forecast horizon in trading days.
    training_cutoff_frac : float
        Fraction of the time range used for training (remainder is validation).

    Returns
    -------
    tuple[TimeSeriesDataSet, TimeSeriesDataSet]
        ``(training, validation)`` datasets.
    """
    # Filter to only columns actually present in the DataFrame
    available_unknown = [c for c in TIME_VARYING_UNKNOWN if c in df.columns]

    # Compute training cutoff based on max time_idx
    max_time_idx = int(df["time_idx"].max())
    training_cutoff = int(max_time_idx * training_cutoff_frac)

    # Build keyword arguments — omit time_varying_known_reals if empty
    kwargs: dict = dict(
        data=df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="Close",
        group_ids=["symbol"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=available_unknown,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer="auto",
    )

    training = TimeSeriesDataSet(**kwargs)

    # Validation set: all data beyond the training cutoff
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=training_cutoff + 1,
    )

    return training, validation
