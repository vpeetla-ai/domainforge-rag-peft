'use client';

import { useMemo, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

const SOLUTIONS = [
  { id: 's0_baseline', label: 'S0 Baseline' },
  { id: 's1_naive_rag', label: 'S1 Naive RAG' },
  { id: 's2_hybrid_rag', label: 'S2 Hybrid RAG' },
  { id: 's3_peft_hybrid', label: 'S3 PEFT + RAG' },
];

type QueryResult = {
  solution: string;
  detected_intent: string;
  chunk_ids: string[];
  triage_json?: string;
  inference_backend: string;
};

type CompareRow = Record<string, string | number>;

export default function HomePage() {
  const [message, setMessage] = useState('Where is my order? I need tracking.');
  const [solution, setSolution] = useState('s1_naive_rag');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [compare, setCompare] = useState<Record<string, CompareRow> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const parsedTriage = useMemo(() => {
    if (!result?.triage_json) return null;
    try {
      return JSON.stringify(JSON.parse(result.triage_json), null, 2);
    } catch {
      return result.triage_json;
    }
  }, [result]);

  async function runQuery() {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_URL}/v1/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, solution }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      setResult(await resp.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Query failed');
    } finally {
      setLoading(false);
    }
  }

  async function runCompare() {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_URL}/v1/eval/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ golden_path: 'data/eval_golden/sample.jsonl' }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      setCompare(await resp.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Compare failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>DomainForge</h1>
      <p className="subtitle">Governed support triage — RAG corpus + PEFT JSON envelope</p>

      <div className="panel">
        <label htmlFor="message">Customer message</label>
        <textarea
          id="message"
          rows={4}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
        <div className="row" style={{ marginTop: '0.75rem' }}>
          <div>
            <label htmlFor="solution">Solution</label>
            <select id="solution" value={solution} onChange={(e) => setSolution(e.target.value)}>
              {SOLUTIONS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          <div style={{ display: 'flex', alignItems: 'end', gap: '0.5rem' }}>
            <button onClick={runQuery} disabled={loading}>
              {loading ? 'Running…' : 'Run query'}
            </button>
            <button className="secondary" onClick={runCompare} disabled={loading}>
              Compare S0–S2
            </button>
          </div>
        </div>
        {error && <p style={{ color: '#fca5a5' }}>{error}</p>}
      </div>

      {result && (
        <div className="panel">
          <div>
            <span className="chip">{result.solution}</span>
            <span className="chip">intent: {result.detected_intent}</span>
            <span className="chip">backend: {result.inference_backend}</span>
          </div>
          <p style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
            Retrieved chunks: {result.chunk_ids.join(', ') || 'none'}
          </p>
          <pre>{parsedTriage}</pre>
        </div>
      )}

      {compare && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Eval compare (golden sample)</h3>
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
              {Object.entries(compare).map(([key, row]) => (
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
    </main>
  );
}
