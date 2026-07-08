"use client";

import { useEffect, useState } from "react";

export type OpsMetrics = {
  service: string;
  total_runs: number;
  success_rate_pct: number;
  p95_latency_ms: number | null;
  active_entities: number;
  slo: { target_uptime_pct: number; success_target_pct: number };
};

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
  const [metrics, setMetrics] = useState<OpsMetrics | null>(null);

  useEffect(() => {
    fetch(metricsUrl, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => data && setMetrics(normalize(data)))
      .catch(() => null);
  }, [metricsUrl]);

  const labels = {
    runs: metricLabels?.runs ?? "Total runs",
    entities: metricLabels?.entities ?? "Active entities",
    latency: metricLabels?.latency ?? "P95 latency",
  };

  return (
    <div className="architect-overview">
      <section>
        <p className="ao-eyebrow">Eagle-eye view</p>
        <h2>Architecture at a glance</h2>
        <p className="ao-lede">{tagline}</p>
        {eagleEyeNote && <p className="ao-note">{eagleEyeNote}</p>}
        {layers.map((layer) => (
          <div key={layer.name} className="architect-layer">
            <span className="architect-tier">{layer.tier}</span>
            <div>
              <strong>{layer.name}</strong>
              <div style={{ fontSize: "0.8rem", color: "var(--muted)" }}>{layer.role}</div>
            </div>
            <div className="architect-chips">
              {layer.components.map((c) => (
                <span key={c} className="architect-chip">{c}</span>
              ))}
            </div>
          </div>
        ))}
      </section>

      <section>
        <p className="ao-eyebrow">Principal tradeoffs</p>
        <h2>Decisions, not defaults</h2>
        <div className="architect-tradeoffs">
          {tradeoffs.map((t) => (
            <div key={t.decision} className="architect-tradeoff">
              <strong>{t.decision}</strong>
              <p><span className="gain">Gain:</span> {t.gain}</p>
              <p><span className="trade">Trade:</span> {t.trade}</p>
            </div>
          ))}
        </div>
      </section>

      {metrics && (
        <section>
          <p className="ao-eyebrow">Production metrics</p>
          <h2>Live from the API</h2>
          <div className="architect-metrics">
            <div className="architect-metric"><span>{labels.runs}</span><strong>{metrics.total_runs}</strong></div>
            <div className="architect-metric"><span>Success rate</span><strong>{metrics.success_rate_pct}%</strong></div>
            <div className="architect-metric"><span>{labels.latency}</span><strong>{metrics.p95_latency_ms ?? "—"}</strong></div>
            <div className="architect-metric"><span>{labels.entities}</span><strong>{metrics.active_entities}</strong></div>
          </div>
        </section>
      )}
    </div>
  );
}

function normalize(data: Record<string, unknown>): OpsMetrics {
  const sloRaw = (data.slo as Record<string, unknown>) || {};
  return {
    service: String(data.service ?? "unknown"),
    total_runs: Number(data.total_runs ?? data.sample_size ?? 0),
    success_rate_pct: Number(data.success_rate_pct ?? 100 - Number(data.failure_rate_pct ?? 0)),
    p95_latency_ms: (data.p95_latency_ms ?? data.p95_ms ?? null) as number | null,
    active_entities: Number(data.active_entities ?? data.invited_users ?? 0),
    slo: {
      target_uptime_pct: Number(sloRaw.target_uptime_pct ?? 99.5),
      success_target_pct: Number(sloRaw.success_target_pct ?? sloRaw.pipeline_success_target_pct ?? 95),
    },
  };
}
