"""Golden eval registry gate for DomainForge triage preference pairs."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from domainforge.eval.alignment import chosen_beats_rejected

try:
    from golden_eval_registry.runner import score_suite
    from golden_eval_registry.schema import parse_manifest
    from golden_eval_registry.validate import load_jsonl

    GOLDEN_EVAL_REGISTRY_AVAILABLE = True
except ImportError:
    GOLDEN_EVAL_REGISTRY_AVAILABLE = False

REGISTRY_PATH = Path(os.getenv("GOLDEN_EVAL_REGISTRY_PATH", "../golden-eval-registry")).resolve()
SUITE_DIR = REGISTRY_PATH / "suites" / "domainforge_triage_preference_v1"


@unittest.skipUnless(
    GOLDEN_EVAL_REGISTRY_AVAILABLE and SUITE_DIR.exists(),
    "golden-eval-registry not available — set GOLDEN_EVAL_REGISTRY_PATH or run in CI",
)
class DomainForgeGoldenEvalGateTests(unittest.TestCase):
    def test_domainforge_triage_preference_v1_suite_passes(self) -> None:
        manifest = parse_manifest(SUITE_DIR / "manifest.json")
        cases = load_jsonl(manifest.cases_path)

        actual_by_id: dict[str, dict] = {}
        for case in cases:
            payload = case["input"]
            chosen = json.dumps(payload["chosen"], separators=(",", ":"))
            rejected = json.dumps(payload["rejected"], separators=(",", ":"))
            beats = chosen_beats_rejected(
                chosen,
                rejected,
                payload["gold_intent"],
                payload.get("allowed_cite_ids", []),
            )
            actual_by_id[str(case["id"])] = {
                "chosen_beats_rejected": beats,
                "detail": "ok" if beats else "alignment scorer did not prefer chosen",
            }

        result = score_suite(manifest, cases, actual_by_id)
        failures = "\n".join(f"{failure.case_id}: {failure.detail}" for failure in result.failures)
        self.assertTrue(result.passed, f"golden eval regressions:\n{failures}")


if __name__ == "__main__":
    unittest.main()
