from __future__ import annotations

import json
from typing import Any

from domainforge.schemas.triage import TriageResponse


def parse_triage_output(raw: str) -> tuple[TriageResponse | None, str | None]:
    try:
        data = json.loads(raw)
        return TriageResponse.model_validate(data), None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def format_adherence_rate(predictions: list[str]) -> float:
    if not predictions:
        return 0.0
    valid = sum(1 for p in predictions if parse_triage_output(p)[0] is not None)
    return 100.0 * valid / len(predictions)
