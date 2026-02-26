"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  AreaSeries,
} from "lightweight-charts";
import type { OHLCVPoint, PredictionPoint } from "@/lib/types";

interface Props {
  historical: OHLCVPoint[];
  predictions: PredictionPoint[];
}

export function CandlestickChart({ historical, predictions }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#a1a1aa",
        fontFamily: "var(--font-mono)",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
      timeScale: { borderColor: "rgba(255,255,255,0.1)" },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    candleSeries.setData(
      historical.map((p) => ({
        time: p.date,
        open: p.open,
        high: p.high,
        low: p.low,
        close: p.close,
      }))
    );

    // 95% confidence band (lighter)
    const band95 = chart.addSeries(AreaSeries, {
      lineColor: "rgba(34, 197, 94, 0.0)",
      topColor: "rgba(34, 197, 94, 0.08)",
      bottomColor: "rgba(34, 197, 94, 0.02)",
      lineWidth: 1 as const,
      lineVisible: false,
    });

    // 80% confidence band (darker)
    const band80 = chart.addSeries(AreaSeries, {
      lineColor: "rgba(34, 197, 94, 0.0)",
      topColor: "rgba(34, 197, 94, 0.15)",
      bottomColor: "rgba(34, 197, 94, 0.05)",
      lineWidth: 1 as const,
      lineVisible: false,
    });

    // Median prediction line
    const medianLine = chart.addSeries(AreaSeries, {
      lineColor: "#22c55e",
      topColor: "rgba(34, 197, 94, 0.0)",
      bottomColor: "rgba(34, 197, 94, 0.0)",
      lineWidth: 2,
      lineStyle: 2,
    });

    if (predictions.length > 0) {
      band95.setData(
        predictions.map((p) => ({ time: p.date, value: p.upper_95 }))
      );
      band80.setData(
        predictions.map((p) => ({ time: p.date, value: p.upper_80 }))
      );
      medianLine.setData(
        predictions.map((p) => ({ time: p.date, value: p.median }))
      );
    }

    chart.timeScale().fitContent();

    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, [historical, predictions]);

  return <div ref={containerRef} className="w-full h-[500px]" />;
}
