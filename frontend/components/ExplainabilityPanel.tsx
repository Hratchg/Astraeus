"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ExplainabilityResponse } from "@/lib/types";

interface Props {
  data: ExplainabilityResponse;
}

export function ExplainabilityPanel({ data }: Props) {
  const importanceData = data.feature_importance
    .slice(0, 10)
    .map((f) => ({
      name: f.feature,
      importance: Math.round(f.importance * 1000) / 10,
    }));

  const maxWeight = Math.max(...data.temporal_attention.map((t) => t.weight));

  return (
    <div className="space-y-6">
      {/* Feature Importance */}
      <div className="bg-card rounded-lg border border-border p-4">
        <h3 className="text-sm font-medium mb-4">
          Feature Importance (Top 10)
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={importanceData}
            layout="vertical"
            margin={{ left: 100, right: 20, top: 5, bottom: 5 }}
          >
            <XAxis
              type="number"
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
              width={95}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(240 10% 5.9%)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "6px",
                color: "#f4f4f5",
                fontSize: 12,
              }}
              formatter={(value) => [`${value}%`, "Importance"]}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {importanceData.map((_, i) => (
                <Cell
                  key={i}
                  fill={`rgba(34, 197, 94, ${0.3 + (0.7 * (importanceData.length - i)) / importanceData.length})`}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Temporal Attention */}
      <div className="bg-card rounded-lg border border-border p-4">
        <h3 className="text-sm font-medium mb-4">
          Temporal Attention (which days mattered most)
        </h3>
        <div className="flex gap-px overflow-x-auto pb-2">
          {data.temporal_attention.map((t, i) => {
            const intensity = maxWeight > 0 ? t.weight / maxWeight : 0;
            return (
              <div
                key={i}
                className="flex flex-col items-center min-w-[14px] group relative"
              >
                <div
                  className="w-3 h-8 rounded-sm"
                  style={{
                    backgroundColor: `rgba(34, 197, 94, ${0.1 + intensity * 0.9})`,
                  }}
                />
                {i % 10 === 0 && (
                  <span className="text-[9px] text-muted-foreground mt-1 -rotate-45 origin-top-left whitespace-nowrap">
                    {t.date.slice(5)}
                  </span>
                )}
                <div className="absolute bottom-full mb-1 hidden group-hover:block bg-popover text-popover-foreground text-[10px] px-2 py-1 rounded border border-border whitespace-nowrap z-10">
                  {t.date}: {(t.weight * 100).toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          Brighter bars indicate days the model paid more attention to when
          generating this forecast. Hover for exact values.
        </p>
      </div>
    </div>
  );
}
