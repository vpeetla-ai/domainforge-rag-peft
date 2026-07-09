"use client";

import { useCallback, useEffect, useState } from "react";

export type OpsMetrics = {
  service: string;
  collected_at?: string;
  total_runs: number;
  success_rate_pct: number;
  p95_latency_ms: number | null;
  active_entities: number;
  slo: { target_uptime_pct: number; success_target_pct: number };
  extra?: Record<string, unknown>;
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

export type AdrLink = {
  title: string;
  href: string;
};

type MetricLabels = {
  runs?: string;
  entities?: string;
  latency?: string;
};

type Props = {
  tagline: string;
  layers: ArchitectLayer[];
  tradeoffs: Tradeoff[];
  metricsUrl: string;
  metricLabels?: MetricLabels;
  eagleEyeNote?: string;
  adrLinks?: AdrLink[];
  docsLinks?: AdrLink[];
};

type MetricsState = "loading" | "live" | "failed";

export function ArchitectOverview({
  tagline,
  layers,
  tradeoffs,
  metricsUrl,
  metricLabels,
  eagleEyeNote,
  adrLinks,
  docsLinks,
}: Props) {
  const [metrics, setMetrics] = useState<OpsMetrics | null>(null);
  const [metricsState, setMetricsState] = useState<MetricsState>("loading");

  const loadMetrics = useCallback(() => {
    setMetricsState("loading");
    fetch(metricsUrl, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data) => {
        setMetrics(normalizeMetrics(data));
        setMetricsState("live");
      })
      .catch(() => {
        setMetrics(null);
        setMetricsState("failed");
      });
  }, [metricsUrl]);

  useEffect(() => {
    loadMetrics();
  }, [loadMetrics]);

  const labels = {
    runs: metricLabels?.runs ?? "Total runs",
    entities: metricLabels?.entities ?? "Active entities",
    latency: metricLabels?.latency ?? "P95 latency",
  };

  const hasDocs = Boolean(adrLinks?.length || docsLinks?.length);

  return (
    <div className="architect-overview">
      <nav className="ao-jump" aria-label="Architecture sections">
        <a href="#ao-stack">Stack</a>
        <a href="#ao-tradeoffs">Tradeoffs</a>
        {hasDocs ? <a href="#ao-adrs">ADRs</a> : null}
        <a href="#ao-metrics">Metrics</a>
      </nav>

      <section id="ao-stack" className="architect-section">
        <p className="ao-eyebrow">Eagle-eye architecture</p>
        <h2 className="ao-title">How the system is wired</h2>
        <p className="ao-lede">{tagline}</p>
        {eagleEyeNote ? <p className="ao-note">{eagleEyeNote}</p> : null}
        {layers.map((layer) => (
          <div key={layer.name} className="architect-layer">
            <span className="architect-tier">{layer.tier}</span>
            <div>
              <strong className="ao-layer-name">{layer.name}</strong>
              <p className="ao-layer-role">{layer.role}</p>
            </div>
            <div className="architect-chips">
              {layer.components.map((c) => (
                <span key={c} className="architect-chip">
                  {c}
                </span>
              ))}
            </div>
          </div>
        ))}
      </section>

      <section id="ao-tradeoffs" className="architect-section">
        <p className="ao-eyebrow">Principal tradeoffs</p>
        <h2 className="ao-title">Decisions with explicit costs</h2>
        <div className="architect-tradeoffs">
          {tradeoffs.map((t) => (
            <div key={t.decision} className="architect-tradeoff">
              <strong className="ao-trade-title">{t.decision}</strong>
              <p>
                <span className="gain">Gain</span> — {t.gain}
              </p>
              <p>
                <span className="trade">Trade</span> — {t.trade}
              </p>
            </div>
          ))}
        </div>
      </section>

      {hasDocs ? (
        <section id="ao-adrs" className="architect-section">
          <p className="ao-eyebrow">Architecture record</p>
          <h2 className="ao-title">ADRs, case studies, and SLOs</h2>
          <ul className="ao-doc-links">
            {adrLinks?.map((link) => (
              <li key={link.href}>
                <a href={link.href} target="_blank" rel="noopener noreferrer">
                  {link.title} →
                </a>
              </li>
            ))}
            {docsLinks?.map((link) => (
              <li key={link.href}>
                <a href={link.href} target="_blank" rel="noopener noreferrer">
                  {link.title} →
                </a>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section id="ao-metrics" className="architect-section">
        <p className="ao-eyebrow">Production metrics</p>
        <h2 className="ao-title">Live operational proof</h2>
        {metricsState === "live" && metrics ? (
          <>
            <div className="architect-metrics">
              <div className="architect-metric">
                <span>{labels.runs}</span>
                <strong>{metrics.total_runs}</strong>
              </div>
              <div className="architect-metric">
                <span>Success rate</span>
                <strong>{metrics.success_rate_pct}%</strong>
              </div>
              <div className="architect-metric">
                <span>{labels.latency}</span>
                <strong>{metrics.p95_latency_ms != null ? `${metrics.p95_latency_ms}ms` : "—"}</strong>
              </div>
              <div className="architect-metric">
                <span>{labels.entities}</span>
                <strong>{metrics.active_entities}</strong>
              </div>
            </div>
            <p className="muted api-hint">
              {metricsUrl.replace(/^https?:\/\/[^/]+/, "")} · SLO {metrics.slo.success_target_pct}% success ·{" "}
              {metrics.slo.target_uptime_pct}% uptime target
            </p>
          </>
        ) : metricsState === "loading" ? (
          <p className="muted">Loading live metrics…</p>
        ) : (
          <div className="ao-metrics-failed">
            <p className="muted">Metrics unavailable — API may be waking from idle (~30s).</p>
            <button type="button" className="secondary" onClick={loadMetrics}>
              Retry
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

function normalizeMetrics(data: Record<string, unknown>): OpsMetrics {
  const sloRaw = (data.slo as Record<string, unknown>) || {};
  const successTarget =
    (sloRaw.success_target_pct as number) ??
    (sloRaw.pipeline_success_target_pct as number) ??
    95.0;

  return {
    service: String(data.service ?? "unknown"),
    collected_at: data.collected_at as string | undefined,
    total_runs: Number(data.total_runs ?? data.sample_size ?? data.total ?? 0),
    success_rate_pct: Number(data.success_rate_pct ?? 100 - Number(data.failure_rate_pct ?? 0)),
    p95_latency_ms:
      (data.p95_latency_ms as number | null) ??
      (data.p95_node_latency_ms as number | null) ??
      (data.p95_ms as number | null) ??
      null,
    active_entities: Number(data.active_entities ?? data.invited_users ?? data.active_users ?? 0),
    slo: {
      target_uptime_pct: Number(sloRaw.target_uptime_pct ?? 99.5),
      success_target_pct: successTarget,
    },
    extra: (data.extra as Record<string, unknown>) ?? undefined,
  };
}
