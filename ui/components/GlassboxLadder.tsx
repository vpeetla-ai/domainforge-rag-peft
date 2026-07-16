"use client";

import { useEffect, useRef, useState } from "react";

export type TraceEvent = {
  name: string;
  attributes?: Record<string, unknown>;
  duration_ms?: number;
};

const NODE_MAP: Record<string, string> = {
  "domain.intent": "intent",
  "domain.retrieve": "retrieve",
  "domain.hybrid_rank": "hybrid",
  "domain.adapter_load": "adapter",
  "domain.dpo_adapter": "dpo",
  "domain.generate": "generate",
  "domain.format_validate": "format",
};

const LADDER = [
  { id: "s0_baseline", label: "S0" },
  { id: "s1_naive_rag", label: "S1" },
  { id: "s2_hybrid_rag", label: "S2" },
  { id: "s3_peft_hybrid", label: "S3" },
  { id: "s4_dpo_peft", label: "S4" },
];

const PILL_STEPS = [
  { id: "intent", label: "Intent" },
  { id: "retrieve", label: "Retrieve" },
  { id: "hybrid", label: "Hybrid" },
  { id: "adapter", label: "PEFT" },
  { id: "dpo", label: "DPO" },
  { id: "generate", label: "Generate" },
  { id: "format", label: "Format" },
];

function gateMessage(ev: TraceEvent): string {
  const a = ev.attributes ?? {};
  if (ev.name === "domain.intent") return `Intent router → ${a.intent ?? "unknown"}`;
  if (ev.name === "domain.retrieve") return `RAG retrieve — ${a.chunk_count ?? 0} chunks for grounding.`;
  if (ev.name === "domain.hybrid_rank") return "Hybrid rank — lexical + semantic fusion.";
  if (ev.name === "domain.adapter_load") return `SFT adapter loaded — ${a.adapter ?? "peft"}.`;
  if (ev.name === "domain.dpo_adapter") return `DPO adapter — preference-aligned format repair.`;
  if (ev.name === "domain.generate") return `Generation via ${a.backend ?? "backend"}.`;
  if (ev.name === "domain.format_validate") return "JSON schema / format adherence check.";
  return ev.name;
}

function skippedNodes(solution: string): Set<string> {
  const skip = new Set<string>();
  if (solution === "s0_baseline") skip.add("retrieve");
  if (solution === "s0_baseline" || solution === "s1_naive_rag") {
    skip.add("hybrid");
  }
  if (solution === "s0_baseline" || solution === "s1_naive_rag" || solution === "s2_hybrid_rag") {
    skip.add("adapter");
    skip.add("dpo");
  }
  if (solution !== "s4_dpo_peft") skip.add("dpo");
  return skip;
}

type Props = {
  solution: string;
  trace: TraceEvent[] | null;
  traceSource: "idle" | "live" | "fallback";
  onMetricsRefresh?: () => void;
};

export function GlassboxLadder({ solution, trace, traceSource, onMetricsRefresh }: Props) {
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [doneNodes, setDoneNodes] = useState<Set<string>>(new Set());
  const [gate, setGate] = useState(
    "S0→S4 ladder — RAG for facts, PEFT/DPO for triage JSON discipline."
  );
  const [eventLog, setEventLog] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setActiveNode(null);
    setDoneNodes(new Set());
    setEventLog([]);

    const skipped = skippedNodes(solution);
    if (!trace?.length) {
      setGate("S0→S4 ladder — RAG for facts, PEFT/DPO for triage JSON discipline.");
      return;
    }

    let i = 0;
    let prevNode: string | null = null;

    const step = () => {
      if (i >= trace.length) {
        if (prevNode) setDoneNodes((prev) => new Set(prev).add(prevNode!));
        setActiveNode(null);
        onMetricsRefresh?.();
        return;
      }
      const ev = trace[i];
      const nodeId = NODE_MAP[ev.name];
      if (nodeId && !skipped.has(nodeId)) {
        if (prevNode) setDoneNodes((prev) => new Set(prev).add(prevNode!));
        setActiveNode(nodeId);
        prevNode = nodeId;
      }
      setGate(gateMessage(ev));
      const ms = ev.duration_ms != null ? ` ${ev.duration_ms}ms` : "";
      setEventLog((prev) => [...prev, `▸ ${ev.name}${ms}`]);
      i += 1;
      timerRef.current = setTimeout(step, traceSource === "fallback" ? 280 : 340);
    };

    step();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- replay on trace/solution change only
  }, [trace, solution, traceSource]);

  const skipped = skippedNodes(solution);
  const totalMs = trace?.reduce((a, e) => a + (e.duration_ms ?? 0), 0) ?? 0;

  return (
    <>
      <div className="gb-center-head">
        <h2>S0→S4 pipeline · trace replay</h2>
        <span
          className={`gb-source-badge${
            traceSource === "live" ? " live" : traceSource === "fallback" ? " fallback" : ""
          }`}
        >
          {traceSource === "live"
            ? "live trace"
            : traceSource === "fallback"
              ? "demo_fallback"
              : "awaiting run"}
        </span>
      </div>

      <div className="gb-ladder-pills">
        {LADDER.map((s) => (
          <span
            key={s.id}
            className={`gb-ladder-step${solution === s.id ? " active" : ""}${doneNodes.size && solution === s.id ? " done" : ""}`}
          >
            {s.label}
          </span>
        ))}
      </div>

      <svg className="gb-pipeline-svg" viewBox="0 0 520 220" role="img" aria-label="Domain adaptation pipeline">
        <defs>
          <marker id="df-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--vp-border-strong)" />
          </marker>
        </defs>
        {[
          { id: "intent", x: 8, y: 88, label: "Intent" },
          { id: "retrieve", x: 88, y: 88, label: "Retrieve" },
          { id: "hybrid", x: 168, y: 88, label: "Hybrid" },
          { id: "adapter", x: 248, y: 88, label: "PEFT" },
          { id: "dpo", x: 328, y: 88, label: "DPO" },
          { id: "generate", x: 408, y: 88, label: "Generate" },
          { id: "format", x: 248, y: 168, label: "Format ✓" },
        ].map((n) => (
          <g
            key={n.id}
            className={`gb-node${skipped.has(n.id) ? " gb-skipped" : ""}${
              activeNode === n.id ? " gb-active" : doneNodes.has(n.id) ? " gb-done" : ""
            }`}
            id={`gb-node-${n.id}`}
          >
            <rect x={n.x} y={n.y} width="72" height="40" rx="8" />
            <text x={n.x + 36} y={n.y + 25} textAnchor="middle">
              {n.label}
            </text>
          </g>
        ))}
        <path className="gb-edge" d="M80 108 H88" markerEnd="url(#df-arrow)" />
        <path className="gb-edge" d="M160 108 H168" markerEnd="url(#df-arrow)" />
        <path className="gb-edge" d="M240 108 H248" markerEnd="url(#df-arrow)" />
        <path className="gb-edge" d="M320 108 H328" markerEnd="url(#df-arrow)" />
        <path className="gb-edge" d="M400 108 H408" markerEnd="url(#df-arrow)" />
        <path className="gb-edge" d="M444 128 V148 H284" markerEnd="url(#df-arrow)" />
      </svg>

      <div className="gb-pills">
        {PILL_STEPS.map((s, i) => (
          <span key={s.id}>
            {i > 0 ? <span className="gb-pill-arrow">→</span> : null}
            <span
              className={`gb-pill${
                activeNode === s.id ? " gb-active" : doneNodes.has(s.id) ? " gb-done" : ""
              }${skipped.has(s.id) ? " gb-skipped-pill" : ""}`}
            >
              {s.label}
            </span>
          </span>
        ))}
      </div>

      <div className="gb-gate">{gate}</div>
      <div className="gb-event-log" aria-live="polite">
        {eventLog.map((line, idx) => (
          <div key={`${line}-${idx}`} className="ev-live">
            {line}
          </div>
        ))}
      </div>
      <div className="gb-ops-strip">
        <span>
          <strong>solution</strong> {solution}
        </span>
        <span>
          <strong>spans</strong> {trace?.length ?? 0}
        </span>
        <span>
          <strong>latency</strong> {totalMs > 0 ? `${totalMs} ms` : "n/a"}
        </span>
      </div>
    </>
  );
}
