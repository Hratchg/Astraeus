"""Build features for all tickers."""
import pandas as pd
from backend.ml.features import prepare_features


def build_all_features(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build features for all tickers, assign time_idx, return combined DataFrame."""
    all_dfs = []
    for symbol, df in ohlcv_data.items():
        featured = prepare_features(df, symbol=symbol)
        if len(featured) > 0:
            all_dfs.append(featured)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs).reset_index(names="date")

    # Assign global time_idx based on date
    unique_dates = sorted(combined["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    combined["time_idx"] = combined["date"].map(date_to_idx)

    return combined
