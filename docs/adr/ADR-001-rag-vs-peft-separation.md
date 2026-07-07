# ADR-001: RAG holds facts; PEFT holds format/behavior

## Status
Accepted

## Context
Customer support triage needs grounded answers from SOP documents and strict JSON output for downstream agents. Full fine-tuning memorizes stale policies; RAG-only models drift on schema and tool grammar.

## Decision
- **RAG corpus** (13 capstone SOPs + chunk index): facts, policies, citation `chunk_id`s
- **PEFT (QLoRA)**: JSON envelope, `intent`, `suggested_action` codes, format discipline
- **Bitext** (~27k pairs): SFT labels only — not copied into vector store as memorization targets

## Consequences
- Two data prep pipelines and manifests to maintain
- Eval must measure faithfulness (citations) and format adherence separately
- Adapter promotion blocked if either regresses

## Links
- Spec: `venkat-ai-portfolio/docs/projects/ENTERPRISE_RAG_PEFT_PIPELINE.md`
- Manifest: `data/manifests/sop_intent_map.json`
