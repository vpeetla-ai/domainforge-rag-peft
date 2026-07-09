"use client";

import { useEffect, useState } from "react";

export type ArchitectLayer = {
  tier: string;
  name: string;
  role: string;
  components: string[];
};

export type Tradeoff = {
  decision: string;
  gain: string;
  trade: string;
};

type Props = {
  tagline: string;
  layers: ArchitectLayer[];
  tradeoffs: Tradeoff[];
  metricsUrl: string;
  metricLabels?: { runs?: string; entities?: string; latency?: string };
  eagleEyeNote?: string;
};

export function ArchitectOverview({
  tagline,
  layers,
  tradeoffs,
  metricsUrl,
  metricLabels,
  eagleEyeNote,
}: Props) {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetch(metricsUrl, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then(setMetrics)
      .catch(() => null);
  }, [metricsUrl]);

  const runs = metrics?.total_runs ?? "—";
  const success = metrics?.success_rate_pct ?? "—";
  const p95 = metrics?.p95_latency_ms ?? metrics?.p95_ms ?? "—";
  const entities = metrics?.active_entities ?? "—";

  return (
    <div className="architect-overview">
      <section className="panel architect-section">
        <p className="ao-eyebrow">Eagle-eye architecture</p>
        <h2 className="ao-title">How the system is wired</h2>
        <p className="ao-lede">{tagline}</p>
        {eagleEyeNote ? <p className="ao-note">{eagleEyeNote}</p> : null}
        {layers.map((layer) => (
          <div key={layer.name} className="architect-layer">
            <span className="architect-tier">{layer.tier}</span>
            <div>
              <strong className="ao-layer-name">{layer.name}</strong>
              <div className="ao-layer-role">{layer.role}</div>
            </div>
            <div className="architect-chips">
              {layer.components.map((c) => (
                <span key={c} className="architect-chip">{c}</span>
              ))}
            </div>
          </div>
        ))}
      </section>

      <section className="panel architect-section">
        <p className="ao-eyebrow">Principal tradeoffs</p>
        <h2 className="ao-title">Decisions with explicit costs</h2>
        <div className="architect-tradeoffs">
          {tradeoffs.map((t) => (
            <div key={t.decision} className="architect-tradeoff">
              <strong className="ao-trade-title">{t.decision}</strong>
              <p><span className="gain">Gain</span> — {t.gain}</p>
              <p><span className="trade">Trade</span> — {t.trade}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="panel architect-section">
        <p className="ao-eyebrow">Production metrics</p>
        <h2 className="ao-title">Live operational proof</h2>
        <div className="architect-metrics">
          <div className="architect-metric"><span>{metricLabels?.runs ?? "Runs"}</span><strong>{String(runs)}</strong></div>
          <div className="architect-metric"><span>Success rate</span><strong>{success}%</strong></div>
          <div className="architect-metric"><span>{metricLabels?.latency ?? "P95"}</span><strong>{p95}{typeof p95 === "number" ? "ms" : ""}</strong></div>
          <div className="architect-metric"><span>{metricLabels?.entities ?? "Entities"}</span><strong>{String(entities)}</strong></div>
        </div>
      </section>
    </div>
  );
}
