"use client";

import type { Horizon } from "@/lib/types";

interface Props {
  value: Horizon;
  onChange: (h: Horizon) => void;
}

const HORIZONS: Horizon[] = [1, 5, 10, 20];
const LABELS: Record<Horizon, string> = {
  1: "1D",
  5: "5D",
  10: "10D",
  20: "20D",
};

export function HorizonToggle({ value, onChange }: Props) {
  return (
    <div className="flex gap-1">
      {HORIZONS.map((h) => (
        <button
          key={h}
          onClick={() => onChange(h)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            value === h
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground"
          }`}
        >
          {LABELS[h]}
        </button>
      ))}
    </div>
  );
}
