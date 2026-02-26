"""Pipeline orchestrator for Astraeus.

Runs the full offline pipeline: fetch OHLCV data, build features,
train TFT model, generate predictions at multiple horizons, and
export results as JSON files.
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.pipeline.fetch_data import fetch_ohlcv, TICKER_UNIVERSE
from backend.pipeline.build_features import build_all_features
from backend.pipeline.train import train_tft
from backend.pipeline.predict import generate_predictions
from backend.pipeline.export import export_predictions, export_explainability, export_metadata
from backend.ml.explain import extract_feature_importance, extract_temporal_attention

logger = logging.getLogger(__name__)

HORIZONS: list[int] = [1, 5, 10, 20]


def run(
    tickers: list[str] | None = None,
    fast: bool = False,
    output_dir: str = "backend/predictions",
) -> None:
    """Run the full Astraeus pipeline.

    Parameters
    ----------
    tickers : list[str], optional
        Ticker symbols to process.  Defaults to all tickers in
        :data:`TICKER_UNIVERSE`.
    fast : bool
        If True, use reduced epochs and smaller model for quick iteration.
    output_dir : str
        Directory for exported JSON prediction files.
    """
    if tickers is None:
        tickers = [t["symbol"] for t in TICKER_UNIVERSE]

    logger.info("Pipeline starting — %d tickers, horizons=%s, fast=%s", len(tickers), HORIZONS, fast)

    # Step 1: Fetch OHLCV data
    logger.info("Step 1/4: Fetching OHLCV data …")
    ohlcv_data: dict = {}
    for symbol in tickers:
        logger.info("  Fetching %s", symbol)
        df = fetch_ohlcv(symbol)
        if not df.empty:
            ohlcv_data[symbol] = df
        else:
            logger.warning("  Skipping %s — no data returned", symbol)

    if not ohlcv_data:
        logger.error("No OHLCV data fetched. Aborting pipeline.")
        return

    logger.info("Fetched %d/%d tickers successfully", len(ohlcv_data), len(tickers))

    # Step 2: Build features
    logger.info("Step 2/4: Building features …")
    feature_df = build_all_features(ohlcv_data)
    if feature_df.empty:
        logger.error("Feature engineering produced empty DataFrame. Aborting.")
        return

    logger.info("Feature DataFrame: %d rows, %d columns", len(feature_df), len(feature_df.columns))

    # Step 3: Train model
    logger.info("Step 3/4: Training TFT model …")
    max_horizon = max(HORIZONS)
    model, training_dataset, trainer = train_tft(
        feature_df,
        max_prediction_length=max_horizon,
        fast=fast,
    )
    logger.info("Training complete.")

    # Step 4: Generate explainability data
    logger.info("Step 4/5: Extracting explainability data …")
    interpretation = None
    try:
        raw_interpretation = model.interpret_output(
            model.predict(
                training_dataset.to_dataloader(batch_size=64, num_workers=0),
                mode="raw",
                return_x=True,
            ),
            reduction="mean",
        )
        interpretation = {
            "encoder_variables": raw_interpretation.get("encoder_variables", []),
            "encoder_importance": raw_interpretation.get("encoder_importance", []),
            "attention": raw_interpretation.get("attention", None),
        }
        logger.info("Explainability extraction complete.")
    except Exception:
        logger.exception("Failed to extract explainability — predictions will still export.")

    # Step 5: Predict + export for each ticker/horizon
    logger.info("Step 5/5: Generating predictions and exporting …")
    processed_tickers = list(ohlcv_data.keys())
    for symbol in processed_tickers:
        for horizon in HORIZONS:
            logger.info("  Predicting %s h=%d", symbol, horizon)
            try:
                pred = generate_predictions(
                    model=model,
                    dataset=training_dataset,
                    historical_df=feature_df,
                    symbol=symbol,
                    horizon=horizon,
                )
                export_predictions(pred, output_dir=output_dir)
                logger.info("  Exported %s_h%d.json", symbol, horizon)
            except Exception:
                logger.exception("  Failed to predict/export %s h=%d", symbol, horizon)

            # Export explainability if extraction succeeded
            if interpretation is not None:
                try:
                    feature_imp = extract_feature_importance(interpretation)
                    sym_df = feature_df[feature_df["symbol"] == symbol] if "symbol" in feature_df.columns else feature_df
                    dates = [str(d)[:10] for d in sym_df.tail(60).index]
                    attn_weights = interpretation.get("attention")
                    temporal_attn = []
                    if attn_weights is not None:
                        import numpy as np
                        attn = np.asarray(attn_weights).mean(axis=0)
                        temporal_attn = extract_temporal_attention(attn[:len(dates)], dates)
                    explain_data = {
                        "symbol": symbol,
                        "horizon_days": horizon,
                        "feature_importance": feature_imp,
                        "temporal_attention": temporal_attn,
                    }
                    export_explainability(explain_data, output_dir=output_dir)
                    logger.info("  Exported %s_explain_h%d.json", symbol, horizon)
                except Exception:
                    logger.exception("  Failed to export explainability for %s h=%d", symbol, horizon)

    # Export metadata
    export_metadata(
        output_dir=output_dir,
        model_version="tft-v1",
        tickers=processed_tickers,
        horizons=HORIZONS,
    )
    logger.info("Pipeline complete. Output in %s", output_dir)


def main() -> None:
    """CLI entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Run the Astraeus prediction pipeline.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols to process (default: all 50 in universe)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced epochs and smaller model for quick iteration",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/predictions",
        help="Directory for exported prediction JSON files (default: backend/predictions)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    run(
        tickers=args.tickers,
        fast=args.fast,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
