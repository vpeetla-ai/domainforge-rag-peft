'use client';

import { useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

type BenchRow = {
  model: string;
  p50_ms: number;
  p95_ms: number;
  tokens_per_sec: number;
  ok_runs: number;
  errors: number;
};

export default function BenchPage() {
  const [rows, setRows] = useState<BenchRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runBench() {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_URL}/v1/bench/ollama`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ runs: 3, models: ['llama3.2:3b', 'mistral:7b'] }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setRows(data.results || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Benchmark failed — is Ollama running?');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="page-hero">
        <p className="eyebrow">Local inference benchmark</p>
        <h1>Ollama bench</h1>
        <p className="subtitle">
          Tokens/sec and P50/P95 latency for structured JSON triage on local Ollama models.
        </p>
      </div>
      <div className="panel">
        <button type="button" onClick={() => void runBench()} disabled={loading}>
          {loading ? 'Running…' : 'Run Ollama benchmark'}
        </button>
        {error && <p className="alert alert-error">{error}</p>}
      </div>
      {rows.length > 0 && (
        <div className="panel">
          <table className="compare-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>P50 ms</th>
                <th>P95 ms</th>
                <th>Tokens/s</th>
                <th>OK / Errors</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.p50_ms}</td>
                  <td>{row.p95_ms}</td>
                  <td>{row.tokens_per_sec}</td>
                  <td>
                    {row.ok_runs} / {row.errors}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
