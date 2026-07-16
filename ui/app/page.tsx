'use client';

import { useMemo, useState } from 'react';
import { GlassboxWorkbench } from '../components/GlassboxWorkbench';
import type { TraceEvent } from '../components/GlassboxLadder';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

const SOLUTIONS = [
  { id: 's0_baseline', label: 'S0 Baseline' },
  { id: 's1_naive_rag', label: 'S1 Naive RAG' },
  { id: 's2_hybrid_rag', label: 'S2 Hybrid RAG' },
  { id: 's3_peft_hybrid', label: 'S3 SFT + RAG' },
  { id: 's4_dpo_peft', label: 'S4 DPO + RAG' },
];

type QueryResult = {
  solution: string;
  detected_intent: string;
  chunk_ids: string[];
  triage_json?: string;
  inference_backend: string;
  trace?: TraceEvent[];
};

type CompareRow = Record<string, string | number>;

type PreferencePair = {
  instruction: string;
  gold_intent: string;
  reject_reason: string;
  chosen: string;
  rejected: string;
};

const WAKE_HINT =
  'The demo API is on Render free tier and may be waking up (~30s after idle). Wait a moment and try again.';

const DEMO_TRACE: TraceEvent[] = [
  { name: 'domain.intent', attributes: { intent: 'track_order' }, duration_ms: 3 },
  { name: 'domain.retrieve', attributes: { chunk_count: 2 }, duration_ms: 12 },
  { name: 'domain.hybrid_rank', duration_ms: 8 },
  { name: 'domain.generate', attributes: { backend: 'mock' }, duration_ms: 14 },
  { name: 'domain.format_validate', duration_ms: 4 },
];

const DEMO_RESULT: QueryResult = {
  solution: 's2_hybrid_rag',
  detected_intent: 'track_order',
  chunk_ids: ['sop-tracking-1', 'sop-tracking-2'],
  inference_backend: 'mock',
  triage_json: JSON.stringify(
    {
      intent: 'track_order',
      confidence: 0.82,
      message: '[demo_fallback] Routed to tracking — order status lookup with grounded SOP context.',
    },
    null,
    2,
  ),
  trace: DEMO_TRACE,
};

function prettyJson(raw: string) {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

async function fetchJson(input: string, init?: RequestInit) {
  let resp: Response;
  try {
    resp = await fetch(input, init);
  } catch {
    throw new Error(WAKE_HINT);
  }
  if (!resp.ok) throw new Error((await resp.text()) || WAKE_HINT);
  return resp.json();
}

const ARCHITECT = {
  layers: [
    { tier: 'L1', name: 'Workbench', role: 'Solution compare UI', components: ['S0–S4 picker', 'Eval harness', 'DPO pairs'] },
    { tier: 'L2', name: 'RAG + adapters', role: 'Grounded triage', components: ['Chroma', 'Intent router', 'PEFT registry'] },
    { tier: 'L3', name: 'Training', role: 'Alignment path', components: ['SFT schema', 'DPO pairs', 'Golden eval'] },
    { tier: 'L4', name: 'Ops', role: 'Corpus proof', components: ['/v1/ops/metrics', 'Security scan', 'SLO'] },
  ],
  tradeoffs: [
    { decision: 'S0→S4 ladder', gain: 'Interview-ready tradeoff narrative per stage', trade: 'More surfaces to maintain than one model' },
    { decision: 'File-based corpus on Render', gain: 'No vector DB bill for demos', trade: 'Cold start rebuilds index' },
    { decision: 'DPO on preference pairs', gain: 'Repairs format without full retrain', trade: 'Needs curated reject examples' },
    { decision: 'Mock LLM default', gain: 'Stable public demo', trade: 'Latency/quality ≠ production inference' },
  ],
  metricsUrl: `${API_URL}/v1/ops/metrics`,
  metricLabels: { runs: 'Corpus chunks', entities: 'Preference pairs', latency: 'P95 latency' },
  adrLinks: [
    { title: 'ADR-019 — DomainForge PEFT ladder', href: 'https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/adr/ADR-019-domainforge-peft-rag-ladder.md' },
    { title: 'ADR-022 — Multi-LoRA serving target', href: 'https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/adr/ADR-022-domainforge-vllm-multi-lora-serving.md' },
  ],
  docsLinks: [
    { title: 'Architecture', href: 'https://github.com/vpeetla-ai/domainforge-rag-peft/blob/main/docs/ARCHITECTURE.md' },
    { title: 'SLO targets', href: 'https://github.com/vpeetla-ai/domainforge-rag-peft/blob/main/docs/SLO.md' },
  ],
};

export default function HomePage() {
  const [message, setMessage] = useState('Where is my order? I need tracking.');
  const [solution, setSolution] = useState('s1_naive_rag');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [trace, setTrace] = useState<TraceEvent[] | null>(null);
  const [traceSource, setTraceSource] = useState<'idle' | 'live' | 'fallback'>('idle');
  const [metricsRefreshToken, setMetricsRefreshToken] = useState(0);
  const [compare, setCompare] = useState<Record<string, CompareRow> | null>(null);
  const [preferences, setPreferences] = useState<PreferencePair[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const parsedTriage = useMemo(() => {
    if (!result?.triage_json) return null;
    return prettyJson(result.triage_json);
  }, [result]);

  const winRate = compare?.s4_vs_s3_preference_win_rate_pct?.value as number | undefined;

  async function runQuery() {
    setLoading(true);
    setError(null);
    setCompare(null);
    setPreferences(null);
    setTrace(null);
    setTraceSource('idle');
    try {
      const data = await fetchJson(`${API_URL}/v1/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, solution }),
      });
      setResult(data);
      setTrace(data.trace ?? []);
      setTraceSource('live');
      setMetricsRefreshToken((t) => t + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Query failed');
      setResult({ ...DEMO_RESULT, solution });
      setTrace(DEMO_TRACE);
      setTraceSource('fallback');
    } finally {
      setLoading(false);
    }
  }

  async function runCompare(solutions?: string[]) {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchJson(`${API_URL}/v1/eval/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ golden_path: 'data/eval_golden/sample.jsonl', solutions }),
      });
      setCompare(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Compare failed');
    } finally {
      setLoading(false);
    }
  }

  async function loadPreferences() {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchJson(`${API_URL}/v1/preferences/samples?limit=4`);
      setPreferences(data.pairs);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Preferences failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <GlassboxWorkbench
      eyebrow="Support triage · RAG + PEFT · glass-box"
      title="Compare S0→S4 on one customer message"
      subtitle="Architecture and metrics stay visible while the center replays pipeline spans from /v1/query — RAG for facts, adapters for JSON schema discipline."
      architect={ARCHITECT}
      solution={solution}
      trace={trace}
      traceSource={traceSource}
      metricsRefreshToken={metricsRefreshToken}
      productPanel={
        <>
          <h2 style={{ margin: '0 0 0.35rem', fontSize: '0.95rem' }}>Support triage</h2>
          <p className="gb-guided">
            <strong>1.</strong> Pick ladder step → <strong>2.</strong> Run query → <strong>3.</strong> Read triage JSON + chunks.
          </p>

          <label htmlFor="message">Customer message</label>
          <textarea id="message" rows={4} value={message} onChange={(e) => setMessage(e.target.value)} />

          <label htmlFor="solution">Solution</label>
          <select id="solution" value={solution} onChange={(e) => setSolution(e.target.value)}>
            {SOLUTIONS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>

          <div className="gb-run-row">
            <button onClick={runQuery} disabled={loading}>
              {loading ? 'Running…' : 'Run triage query'}
            </button>
          </div>

          {error && <p className="alert alert-error">{error}</p>}

          {!result && !loading && !error && (
            <p className="muted microcopy" style={{ marginTop: '0.75rem' }}>
              Results appear here — intent, retrieved chunks, and triage JSON for the selected step.
            </p>
          )}

          {result && (
            <div style={{ marginTop: '0.85rem' }}>
              <div>
                <span className="chip">{result.solution}</span>
                <span className="chip">intent: {result.detected_intent}</span>
                <span className="chip">backend: {result.inference_backend}</span>
              </div>
              <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
                Chunks: {result.chunk_ids.join(', ') || 'none'}
              </p>
              <pre style={{ maxHeight: 220, fontSize: '0.78rem' }}>{parsedTriage}</pre>
            </div>
          )}
        </>
      }
      secondaryPanel={
        <>
          <details>
            <summary>Eval tools — compare ladder · DPO preference pairs</summary>
            <div className="panel" style={{ marginTop: '0.65rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button className="secondary" onClick={() => runCompare()} disabled={loading}>
                  Compare S0–S2
                </button>
                <button
                  className="secondary"
                  onClick={() => runCompare(['s3_peft_hybrid', 's4_dpo_peft'])}
                  disabled={loading}
                >
                  Compare S3 vs S4
                </button>
                <button className="secondary" onClick={loadPreferences} disabled={loading}>
                  View preference pairs
                </button>
              </div>
            </div>
          </details>

          {compare && (
            <div className="panel">
              <h3 style={{ marginTop: 0 }}>Eval compare (golden sample)</h3>
              {winRate !== undefined && <p className="chip">S4 preference win-rate vs S3: {winRate}%</p>}
              <table className="compare-table">
                <thead>
                  <tr>
                    <th>Solution</th>
                    <th>Format %</th>
                    <th>Intent %</th>
                    <th>Hallucination %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(compare)
                    .filter(([key]) => !key.includes('preference_win_rate'))
                    .map(([key, row]) => (
                      <tr key={key}>
                        <td>{key}</td>
                        <td>{row.format_adherence_pct}</td>
                        <td>{row.intent_accuracy_pct}</td>
                        <td>{row.hallucination_freq_pct}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}

          {preferences && (
            <div className="panel">
              <h3 style={{ marginTop: 0 }}>DPO preference pairs (chosen vs rejected)</h3>
              {preferences.map((pair, idx) => (
                <div key={`${pair.instruction}-${idx}`} className="pref-card">
                  <p>
                    <strong>{pair.instruction}</strong>
                  </p>
                  <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
                    gold intent: {pair.gold_intent} · reject reason: {pair.reject_reason}
                  </p>
                  <div className="pref-grid">
                    <div>
                      <h4>Chosen ✓</h4>
                      <pre>{prettyJson(pair.chosen)}</pre>
                    </div>
                    <div>
                      <h4>Rejected ✗</h4>
                      <pre className="rejected">{prettyJson(pair.rejected)}</pre>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      }
    />
  );
}
