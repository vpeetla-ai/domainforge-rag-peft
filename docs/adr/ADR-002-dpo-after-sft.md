# ADR-002: DPO After SFT for Triage Alignment

## Status
Accepted

## Context
SFT (QLoRA) teaches the JSON envelope and intent grammar, but models still produce **plausible wrong** outputs: wrong intent on adversarial prompts, invented `cite_faq_ids`, or incorrect `suggested_action` escalation.

## Decision
- Add **DPO (Direct Preference Optimization)** as solution **S4** on the behavior plane only
- Preference pairs: `(prompt, chosen, rejected)` labeled by governed scorers — not human thumbs
- **Rejected** outputs are hard negatives: wrong intent, hallucinated cites, wrong action
- **SFT adapter (S3)** is warm-start; **DPO adapter (S4)** promoted only if alignment metrics improve
- Promotion gate: S4 must beat S3 on `preference_win_rate_pct` without format regression

## Consequences
- Third training CLI path: `domainforge-train dpo`
- Preference data prep: `domainforge-prep build-preferences`
- Eval ladder extends to S0→S4; UI shows chosen vs rejected pairs
- GPU required for production DPO on Mistral; CPU smoke uses `tiny-gpt2`

## Links
- Canonical ADR: [ADR-020](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/adr/ADR-020-dpo-after-sft-alignment.md)
- Related: [ADR-001](ADR-001-rag-vs-peft-separation.md) · [ADR-019](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/adr/ADR-019-rag-facts-peft-behavior.md)
