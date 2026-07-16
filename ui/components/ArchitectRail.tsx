"use client";

import { useCallback, useEffect, useState } from "react";
import type { ArchitectLayer, AdrLink, Tradeoff } from "./ArchitectOverview";

type MetricLabels = {
  runs?: string;
  entities?: string;
  latency?: string;
};

type Props = {
  layers: ArchitectLayer[];
  tradeoffs: Tradeoff[];
  metricsUrl: string;
  metricLabels?: MetricLabels;
  adrLinks?: AdrLink[];
  docsLinks?: AdrLink[];
  refreshToken?: number;
};

type MetricsState = "loading" | "live" | "failed";

export function ArchitectRail({
  layers,
  tradeoffs,
  metricsUrl,
  metricLabels,
  adrLinks,
  docsLinks,
  refreshToken = 0,
}: Props) {
  const [metricsState, setMetricsState] = useState<MetricsState>("loading");
  const [metrics, setMetrics] = useState<Record<string, number | string | null>>({});

  const loadMetrics = useCallback(() => {
    setMetricsState("loading");
    fetch(metricsUrl, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data) => {
        setMetrics({
          runs: Number(data.total_runs ?? data.corpus_chunks ?? 0),
          success: Number(data.success_rate_pct ?? 100),
          latency: (data.p95_latency_ms as number | null) ?? null,
          entities: Number(data.active_entities ?? 0),
        });
        setMetricsState("live");
      })
      .catch(() => setMetricsState("failed"));
  }, [metricsUrl]);

  useEffect(() => {
    loadMetrics();
  }, [loadMetrics, refreshToken]);

  const labels = {
    runs: metricLabels?.runs ?? "Corpus chunks",
    entities: metricLabels?.entities ?? "Preference pairs",
    latency: metricLabels?.latency ?? "P95",
  };

  const links = [...(adrLinks ?? []), ...(docsLinks ?? [])].slice(0, 4);

  return (
    <>
      <h2 className="gb-rail-title">Stack</h2>
      <div className="gb-stack">
        {layers.map((layer) => (
          <div key={layer.name} className="gb-stack-layer">
            <div className="gb-stack-tier">{layer.tier}</div>
            <div className="gb-stack-name">{layer.name}</div>
            <div className="gb-stack-role">{layer.role}</div>
          </div>
        ))}
      </div>

      <h2 className="gb-rail-title">Live metrics</h2>
      {metricsState === "live" ? (
        <div className="gb-metrics">
          <div className="gb-metric">
            <span>{labels.runs}</span>
            <strong>{metrics.runs}</strong>
          </div>
          <div className="gb-metric">
            <span>Success</span>
            <strong>{metrics.success}%</strong>
          </div>
          <div className="gb-metric">
            <span>{labels.latency}</span>
            <strong>{metrics.latency != null ? `${metrics.latency}ms` : "—"}</strong>
          </div>
          <div className="gb-metric">
            <span>{labels.entities}</span>
            <strong>{metrics.entities}</strong>
          </div>
        </div>
      ) : metricsState === "loading" ? (
        <p className="muted" style={{ fontSize: "0.78rem" }}>
          Loading…
        </p>
      ) : (
        <div className="gb-metrics-failed">
          <p className="muted" style={{ fontSize: "0.78rem", margin: 0 }}>
            API waking (~30s)…
          </p>
          <button type="button" className="secondary" onClick={loadMetrics}>
            Retry
          </button>
        </div>
      )}

      <h2 className="gb-rail-title">Tradeoffs</h2>
      {tradeoffs.slice(0, 2).map((t) => (
        <div key={t.decision} className="gb-tradeoff">
          <strong>{t.decision}</strong>
          <p>{t.gain}</p>
        </div>
      ))}

      {links.length > 0 ? (
        <>
          <h2 className="gb-rail-title">ADRs</h2>
          <ul className="gb-adr-links">
            {links.map((link) => (
              <li key={link.href}>
                <a href={link.href} target="_blank" rel="noopener noreferrer">
                  {link.title} →
                </a>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </>
  );
}

export type { Props as ArchitectRailProps };
