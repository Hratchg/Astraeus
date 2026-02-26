import pytest
import numpy as np
from backend.ml.explain import extract_feature_importance, extract_temporal_attention


def test_extract_feature_importance_returns_sorted_list():
    mock_interpretation = {
        "encoder_variables": ["Close", "RSI_14", "Volume", "ATR_14"],
        "encoder_importance": [0.15, 0.30, 0.10, 0.25],
    }
    result = extract_feature_importance(mock_interpretation)
    assert len(result) == 4
    assert result[0]["importance"] >= result[1]["importance"]  # Sorted descending
    assert "feature" in result[0]
    assert "importance" in result[0]


def test_extract_temporal_attention_returns_weights():
    mock_attention = np.random.dirichlet(np.ones(60))  # 60 timesteps
    dates = [f"2026-02-{i:02d}" for i in range(1, 61)]
    result = extract_temporal_attention(mock_attention, dates)
    assert len(result) == 60
    assert "date" in result[0]
    assert "weight" in result[0]
    assert abs(sum(r["weight"] for r in result) - 1.0) < 1e-4
