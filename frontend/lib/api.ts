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
    const body = await res
      .json()
      .catch(() => ({ detail: res.statusText }));
    // FastAPI wraps HTTPException detail in {"detail": ...}
    const detail = body.detail;
    const message =
      typeof detail === "object" && detail !== null
        ? detail.detail || detail.error
        : detail || "API error";
    throw new Error(message);
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
