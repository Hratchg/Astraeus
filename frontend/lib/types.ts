export interface TickerInfo {
  symbol: string;
  name: string;
  sector: string;
  confidence_level: "high" | "low";
}

export interface OHLCVPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionPoint {
  date: string;
  median: number;
  lower_80: number;
  upper_80: number;
  lower_95: number;
  upper_95: number;
}

export interface PredictionResponse {
  symbol: string;
  horizon_days: number;
  generated_at: string;
  model_version: string;
  historical: OHLCVPoint[];
  predictions: PredictionPoint[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface TemporalAttention {
  date: string;
  weight: number;
}

export interface ExplainabilityResponse {
  symbol: string;
  horizon_days: number;
  feature_importance: FeatureImportance[];
  temporal_attention: TemporalAttention[];
}

export type Horizon = 1 | 5 | 10 | 20;
