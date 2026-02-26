"""Pydantic response models matching API contract."""
from pydantic import BaseModel


class TickerInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    confidence_level: str  # "high" or "low"


class TickerListResponse(BaseModel):
    tickers: list[TickerInfo]


class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PredictionPoint(BaseModel):
    date: str
    median: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class PredictionResponse(BaseModel):
    symbol: str
    horizon_days: int
    generated_at: str
    model_version: str
    historical: list[OHLCVPoint]
    predictions: list[PredictionPoint]


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class TemporalAttention(BaseModel):
    date: str
    weight: float


class ExplainabilityResponse(BaseModel):
    symbol: str
    horizon_days: int
    feature_importance: list[FeatureImportance]
    temporal_attention: list[TemporalAttention]


class ErrorDetail(BaseModel):
    error: str
    detail: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
