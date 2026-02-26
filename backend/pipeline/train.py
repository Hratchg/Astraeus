"""Train TFT model."""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from backend.ml.dataset import build_time_series_dataset
from backend.ml.model import create_tft_model, TFTConfig
import pandas as pd


def train_tft(
    feature_df: pd.DataFrame,
    max_prediction_length: int = 20,
    max_epochs: int = 30,
    batch_size: int = 64,
    config: TFTConfig | None = None,
    fast: bool = False,
) -> tuple:
    """Train TFT model and return (model, training_dataset, trainer)."""
    if fast:
        max_epochs = 2
        if config is None:
            config = TFTConfig(hidden_size=16, attention_head_size=1)

    training, validation = build_time_series_dataset(
        feature_df, max_prediction_length=max_prediction_length
    )

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, num_workers=0)

    model = create_tft_model(training, config)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        best_model = type(model).load_from_checkpoint(best_model_path)
    else:
        best_model = model

    return best_model, training, trainer
