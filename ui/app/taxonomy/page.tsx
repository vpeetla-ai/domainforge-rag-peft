import Link from "next/link";

const ADAPTATION = [
  { label: "Regular / base LM", where: "S0 baseline", status: "Live workbench" },
  { label: "RAG (S1/S2)", where: "Facts plane", status: "Live — Chroma + hybrid" },
  { label: "QLoRA SFT (S3)", where: "PEFT train CLI", status: "Live — GPU path documented" },
  { label: "DPO (S4)", where: "Preference pairs", status: "Live — win-rate compare" },
  { label: "Multi-LoRA serve", where: "vLLM Lab Path B", status: "Educational — ADR-022" },
];

const TASKS = [
  { label: "Intent classification (~27 intents)", plane: "LLM JSON triage", note: "Not sklearn tabular" },
  { label: "Structured JSON / schema", plane: "S3/S4 + golden-eval", note: "format_validate span" },
  { label: "RAG-grounded facts", plane: "S1/S2 retrieve", note: "Separate from PEFT weights" },
];

export default function TaxonomyPage() {
  return (
    <div className="gb-shell">
      <div className="gb-hero page-hero">
        <p className="eyebrow">DomainForge · taxonomy slice</p>
        <h1>Train plane — S0→S4 + adaptation methods</h1>
        <p className="lede">
          DomainForge owns the solution ladder and QLoRA/DPO training path. ModelForge owns posture,
          CUDA receipts, and the full cross-product taxonomy.
        </p>
        <p className="gb-guided" style={{ marginTop: "0.75rem" }}>
          <Link href="/">← Back to workbench</Link>
          {" · "}
          <a href="https://modelforge-gamma.vercel.app/taxonomy" target="_blank" rel="noreferrer">
            ModelForge taxonomy tab ↗
          </a>
          {" · "}
          <a href="https://venkat-ai.com/model-plane" target="_blank" rel="noreferrer">
            venkat-ai.com/model-plane ↗
          </a>
        </p>
      </div>

      <section className="gb-secondary" style={{ display: "grid", gap: "1.25rem" }}>
        <div className="gb-product">
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Adaptation methods (DomainForge scope)</h2>
          <table style={{ width: "100%", fontSize: "0.88rem", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "0.4rem 0" }}>Method</th>
                <th style={{ textAlign: "left" }}>Where</th>
                <th style={{ textAlign: "left" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {ADAPTATION.map((row) => (
                <tr key={row.label} style={{ borderTop: "1px solid var(--vp-border)" }}>
                  <td style={{ padding: "0.5rem 0" }}>{row.label}</td>
                  <td>{row.where}</td>
                  <td>{row.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="gb-product">
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Task types trained here</h2>
          <ul style={{ margin: 0, paddingLeft: "1.1rem", lineHeight: 1.7 }}>
            {TASKS.map((t) => (
              <li key={t.label}>
                <strong>{t.label}</strong> — {t.plane}. <em>{t.note}</em>
              </li>
            ))}
          </ul>
        </div>

        <div className="gb-product">
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Classical ML stack (org lane)</h2>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.6, color: "var(--vp-text-secondary)" }}>
            Tabular classification/regression, registry, skew, and ML CI/CD are interview-playbook
            depth — not a live sklearn service in this repo. See the{" "}
            <a href="https://venkat-ai.com/model-plane">full taxonomy</a> classical ML tab for
            mlops-llmops study links and honesty labels.
          </p>
        </div>
      </section>
    </div>
  );
}
