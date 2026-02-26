"use client";

import { useState, useEffect, use } from "react";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { ExplainabilityPanel } from "@/components/ExplainabilityPanel";
import { HorizonToggle } from "@/components/HorizonToggle";
import { getPredictions, getExplainability } from "@/lib/api";
import type {
  PredictionResponse,
  ExplainabilityResponse,
  Horizon,
} from "@/lib/types";

export default function TickerPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explain, setExplain] = useState<ExplainabilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"chart" | "explain">("chart");

  useEffect(() => {
    setLoading(true);
    setError(null);
    setExplain(null);

    getPredictions(symbol, horizon)
      .then((pred) => setPrediction(pred))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));

    getExplainability(symbol, horizon)
      .then((exp) => setExplain(exp))
      .catch(() => setExplain(null));
  }, [symbol, horizon]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading predictions...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-destructive font-medium mb-1">
            Failed to load data
          </p>
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  if (!prediction) return null;

  const lastClose =
    prediction.historical[prediction.historical.length - 1]?.close;
  const medianPred = prediction.predictions[prediction.predictions.length - 1]?.median;
  const pctChange = lastClose && medianPred
    ? (((medianPred - lastClose) / lastClose) * 100).toFixed(2)
    : null;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight font-mono">
            {symbol}
          </h2>
          <div className="flex items-center gap-3 mt-1">
            {lastClose && (
              <span className="text-lg font-mono">${lastClose.toFixed(2)}</span>
            )}
            {pctChange && (
              <span
                className={`text-sm font-mono ${
                  parseFloat(pctChange) >= 0
                    ? "text-green-500"
                    : "text-red-500"
                }`}
              >
                {parseFloat(pctChange) >= 0 ? "+" : ""}
                {pctChange}% ({horizon}D forecast)
              </span>
            )}
          </div>
        </div>
        <HorizonToggle value={horizon} onChange={setHorizon} />
      </div>

      {/* Tab toggle */}
      <div className="flex gap-1 border-b border-border">
        <button
          onClick={() => setTab("chart")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === "chart"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Price Chart
        </button>
        <button
          onClick={() => setTab("explain")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === "explain"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          Explainability
        </button>
      </div>

      {/* Content */}
      {tab === "chart" && (
        <div className="bg-card rounded-lg border border-border p-4">
          <CandlestickChart
            historical={prediction.historical}
            predictions={prediction.predictions}
          />
          <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-green-500 inline-block" /> Median
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-green-500/15 inline-block rounded-sm" />{" "}
              80% CI
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-green-500/8 inline-block rounded-sm" />{" "}
              95% CI
            </span>
            <span className="ml-auto">
              Model: {prediction.model_version} | Generated:{" "}
              {new Date(prediction.generated_at).toLocaleDateString()}
            </span>
          </div>
        </div>
      )}

      {tab === "explain" && (
        explain ? (
          <ExplainabilityPanel data={explain} />
        ) : (
          <div className="flex items-center justify-center py-12">
            <p className="text-sm text-muted-foreground">
              Explainability data is not available for this ticker.
            </p>
          </div>
        )
      )}
    </div>
  );
}
