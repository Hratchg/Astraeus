"use client";

import { TickerSearch } from "@/components/TickerSearch";

export function ClientShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <TickerSearch />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
