'use client';

import { useMemo, useState } from 'react';
import { ArchitectOverview } from '../components/ArchitectOverview';
import { ProductWorkbench } from '../components/ProductWorkbench';

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
};

type CompareRow = Record<string, string | number>;

type PreferencePair = {
  instruction: string;
  gold_intent: string;
  reject_reason: string;
  chosen: string;
  rejected: string;
  context_blocks?: string[];
};

function prettyJson(raw: string) {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

const WAKE_HINT =
  'The demo API is on Render free tier and may be waking up (~30s after idle). Wait a moment and try again.';

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

export default function HomePage() {
  const [message, setMessage] = useState('Where is my order? I need tracking.');
  const [solution, setSolution] = useState('s1_naive_rag');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
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
    try {
      const data = await fetchJson(`${API_URL}/v1/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, solution }),
      });
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Query failed');
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
    <ProductWorkbench
      eyebrow="Support triage · RAG + PEFT"
      productName="DomainForge"
      subtitle="Compare S0→S4 on the same customer message — RAG for facts, adapters for JSON schema discipline."
      productPanel={
        <>
          <div className="panel">
            <label htmlFor="message">Customer message</label>
            <textarea id="message" rows={4} value={message} onChange={(e) => setMessage(e.target.value)} />
            <div className="row" style={{ marginTop: '0.75rem' }}>
              <div>
                <label htmlFor="solution">Solution</label>
                <select id="solution" value={solution} onChange={(e) => setSolution(e.target.value)}>
                  {SOLUTIONS.map((s) => (
                    <option key={s.id} value={s.id}>{s.label}</option>
                  ))}
                </select>
              </div>
              <div style={{ display: 'flex', alignItems: 'end', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button onClick={runQuery} disabled={loading}>{loading ? 'Running…' : 'Run query'}</button>
                <button className="secondary" onClick={() => runCompare()} disabled={loading}>Compare S0–S2</button>
                <button className="secondary" onClick={() => runCompare(['s3_peft_hybrid', 's4_dpo_peft'])} disabled={loading}>Compare S3 vs S4</button>
                <button className="secondary" onClick={loadPreferences} disabled={loading}>View preference pairs</button>
              </div>
            </div>
            {error && <p className="alert alert-error">{error}</p>}
          </div>

          {result && (
            <div className="panel">
              <div>
                <span className="chip">{result.solution}</span>
                <span className="chip">intent: {result.detected_intent}</span>
                <span className="chip">backend: {result.inference_backend}</span>
              </div>
              <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Retrieved chunks: {result.chunk_ids.join(', ') || 'none'}</p>
              <pre>{parsedTriage}</pre>
            </div>
          )}

          {compare && (
            <div className="panel">
              <h3 style={{ marginTop: 0 }}>Eval compare (golden sample)</h3>
              {winRate !== undefined && <p className="chip">S4 preference win-rate vs S3: {winRate}%</p>}
              <table className="compare-table">
                <thead>
                  <tr><th>Solution</th><th>Format %</th><th>Intent %</th><th>Hallucination %</th></tr>
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
                  <p><strong>{pair.instruction}</strong></p>
                  <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
                    gold intent: {pair.gold_intent} · reject reason: {pair.reject_reason}
                  </p>
                  <div className="pref-grid">
                    <div><h4>Chosen ✓</h4><pre>{prettyJson(pair.chosen)}</pre></div>
                    <div><h4>Rejected ✗</h4><pre className="rejected">{prettyJson(pair.rejected)}</pre></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      }
      architecturePanel={
        <ArchitectOverview
          tagline="Domain adaptation ladder: baseline → naive RAG → hybrid → SFT → DPO. Each step trades complexity for format compliance and grounding."
          layers={[
            { tier: 'L1', name: 'Workbench', role: 'Solution compare UI', components: ['S0–S4 picker', 'Eval harness', 'DPO pairs'] },
            { tier: 'L2', name: 'RAG + adapters', role: 'Grounded triage', components: ['Chroma', 'Intent router', 'PEFT registry'] },
            { tier: 'L3', name: 'Training', role: 'Alignment path', components: ['SFT schema', 'DPO pairs', 'Golden eval'] },
            { tier: 'L4', name: 'Ops', role: 'Corpus proof', components: ['/v1/ops/metrics', 'Security scan', 'SLO'] },
          ]}
          tradeoffs={[
            { decision: 'S0→S4 ladder', gain: 'Interview-ready tradeoff narrative per stage', trade: 'More surfaces to maintain than one model' },
            { decision: 'File-based corpus on Render', gain: 'No vector DB bill for demos', trade: 'Cold start rebuilds index' },
            { decision: 'DPO on preference pairs', gain: 'Repairs format without full retrain', trade: 'Needs curated reject examples' },
            { decision: 'Mock LLM default', gain: 'Stable public demo', trade: 'Latency/quality ≠ production inference' },
          ]}
          metricsUrl={`${API_URL}/v1/ops/metrics`}
          metricLabels={{ runs: 'Corpus chunks', entities: 'Preference pairs', latency: 'P95 latency' }}
        />
      }
    />
  );
}
