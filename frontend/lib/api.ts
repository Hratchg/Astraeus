import type {
  TickerInfo,
  PredictionResponse,
  ExplainabilityResponse,
  Horizon,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const error = await res
      .json()
      .catch(() => ({ error: "unknown", detail: res.statusText }));
    throw new Error(error.detail || error.error || "API error");
  }
  return res.json();
}

export async function getTickers(query?: string): Promise<TickerInfo[]> {
  const params = query ? `?q=${encodeURIComponent(query)}` : "";
  const data = await fetchJSON<{ tickers: TickerInfo[] }>(`/tickers${params}`);
  return data.tickers;
}

export async function getPredictions(
  symbol: string,
  horizon: Horizon
): Promise<PredictionResponse> {
  return fetchJSON<PredictionResponse>(
    `/predictions/${symbol}?horizon=${horizon}`
  );
}

export async function getExplainability(
  symbol: string,
  horizon: Horizon
): Promise<ExplainabilityResponse> {
  return fetchJSON<ExplainabilityResponse>(
    `/explainability/${symbol}?horizon=${horizon}`
  );
}
