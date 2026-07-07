from __future__ import annotations

import json


def intent_accuracy_rate(predictions: list[str], gold_intents: list[str]) -> float:
    if not predictions or len(predictions) != len(gold_intents):
        return 0.0
    correct = 0
    for pred, gold in zip(predictions, gold_intents, strict=True):
        try:
            data = json.loads(pred)
            if data.get("intent") == gold:
                correct += 1
        except json.JSONDecodeError:
            continue
    return 100.0 * correct / len(predictions)
