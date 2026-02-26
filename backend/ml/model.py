"""
TFT model wrapper for Astraeus.

Creates a TemporalFusionTransformer configured for quantile forecasting
with 80% and 95% confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: list[float] = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
"""Quantiles for 80% (0.1–0.9) and 95% (0.025–0.975) confidence intervals."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TFTConfig:
    """Hyperparameters for the Temporal Fusion Transformer."""

    learning_rate: float = 0.001
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    output_size: int = 7  # one per quantile


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def create_tft_model(
    training_dataset: TimeSeriesDataSet,
    config: TFTConfig | None = None,
) -> TemporalFusionTransformer:
    """Create a TFT model from a training dataset and optional config.

    Parameters
    ----------
    training_dataset : TimeSeriesDataSet
        The training dataset (used to infer input dimensions).
    config : TFTConfig, optional
        Model hyperparameters.  Defaults to :class:`TFTConfig` defaults.

    Returns
    -------
    TemporalFusionTransformer
        An untrained model ready for ``pytorch_lightning.Trainer.fit()``.
    """
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
    )

    return model
