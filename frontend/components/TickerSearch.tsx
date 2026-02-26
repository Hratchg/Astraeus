"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getTickers } from "@/lib/api";
import type { TickerInfo } from "@/lib/types";

export function TickerSearch() {
  const [query, setQuery] = useState("");
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const pathname = usePathname();

  // Extract current symbol from pathname like /ticker/AAPL
  const activeSymbol = pathname?.startsWith("/ticker/")
    ? pathname.split("/")[2]
    : null;

  useEffect(() => {
    const timeout = setTimeout(() => {
      setLoading(true);
      setError(false);
      getTickers(query || undefined)
        .then(setTickers)
        .catch(() => {
          setTickers([]);
          setError(true);
        })
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(timeout);
  }, [query]);

  const sectors = [...new Set(tickers.map((t) => t.sector))];

  return (
    <div className="w-64 border-r border-border flex flex-col h-full">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-bold tracking-tight mb-3">Astraeus</h1>
        <Input
          placeholder="Search tickers..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="bg-muted"
        />
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2">
          {loading && tickers.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-4">
              Loading tickers...
            </p>
          )}
          {!loading && error && (
            <p className="text-xs text-destructive px-2 py-4">
              Failed to load tickers. Is the API running?
            </p>
          )}
          {!loading && !error && tickers.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-4">
              No tickers found.
            </p>
          )}
          {sectors.map((sector) => (
            <div key={sector} className="mb-4">
              <p className="text-xs font-medium text-muted-foreground px-2 mb-1">
                {sector}
              </p>
              {tickers
                .filter((t) => t.sector === sector)
                .map((t) => {
                  const isActive = activeSymbol === t.symbol;
                  return (
                    <Link
                      key={t.symbol}
                      href={`/ticker/${t.symbol}`}
                      className={`w-full text-left px-2 py-1.5 rounded-md flex items-center justify-between group ${
                        isActive
                          ? "bg-muted text-foreground"
                          : "hover:bg-muted"
                      }`}
                    >
                      <div>
                        <span className="font-mono text-sm font-medium">
                          {t.symbol}
                        </span>
                        <span className="text-xs text-muted-foreground ml-2">
                          {t.name}
                        </span>
                      </div>
                      {t.confidence_level === "low" && (
                        <Badge
                          variant="outline"
                          className="text-amber-500 border-amber-500 text-[10px]"
                        >
                          Low
                        </Badge>
                      )}
                    </Link>
                  );
                })}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
