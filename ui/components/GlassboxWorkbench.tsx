"use client";

import type { ReactNode } from "react";
import { ArchitectRail, type ArchitectRailProps } from "./ArchitectRail";
import { GlassboxLadder, type TraceEvent } from "./GlassboxLadder";

type Props = {
  eyebrow: string;
  title: string;
  subtitle: string;
  architect: ArchitectRailProps;
  solution: string;
  trace: TraceEvent[] | null;
  traceSource: "idle" | "live" | "fallback";
  productPanel: ReactNode;
  secondaryPanel?: ReactNode;
  metricsRefreshToken?: number;
};

export function GlassboxWorkbench({
  eyebrow,
  title,
  subtitle,
  architect,
  solution,
  trace,
  traceSource,
  productPanel,
  secondaryPanel,
  metricsRefreshToken = 0,
}: Props) {
  return (
    <div className="gb-shell">
      <div className="gb-hero page-hero">
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="lede">{subtitle}</p>
      </div>

      <div className="gb-workbench">
        <aside className="gb-rail" aria-label="Architecture and metrics">
          <ArchitectRail {...architect} refreshToken={metricsRefreshToken} />
        </aside>

        <section className="gb-center" aria-label="S0 to S4 pipeline glass-box">
          <GlassboxLadder solution={solution} trace={trace} traceSource={traceSource} />
        </section>

        <aside className="gb-product" aria-label="Support triage product">
          {productPanel}
        </aside>
      </div>

      {secondaryPanel ? <div className="gb-secondary">{secondaryPanel}</div> : null}
    </div>
  );
}
