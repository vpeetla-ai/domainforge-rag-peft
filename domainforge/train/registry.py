from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"adapters": []}


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def register_adapter(
    registry_path: Path,
    adapter_id: str,
    adapter_dir: Path,
    base_model: str,
    status: str = "candidate",
    eval_scores: dict[str, Any] | None = None,
    training_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    entry = {
        "id": adapter_id,
        "path": str(adapter_dir),
        "base_model": base_model,
        "status": status,
        "eval_scores": eval_scores or {},
        "training_manifest": training_manifest or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["adapters"] = [a for a in registry.get("adapters", []) if a.get("id") != adapter_id]
    registry["adapters"].append(entry)
    save_registry(registry_path, registry)
    return entry


def get_adapter(registry_path: Path, adapter_id: str) -> dict[str, Any] | None:
    registry = load_registry(registry_path)
    for adapter in registry.get("adapters", []):
        if adapter.get("id") == adapter_id:
            return adapter
    return None


def promote_adapter(registry_path: Path, adapter_id: str) -> dict[str, Any]:
    registry = load_registry(registry_path)
    promoted: dict[str, Any] | None = None
    for adapter in registry.get("adapters", []):
        if adapter.get("id") == adapter_id:
            adapter["status"] = "promoted"
            promoted = adapter
        elif adapter.get("status") == "promoted":
            adapter["status"] = "retired"
    if promoted is None:
        raise ValueError(f"Adapter not found: {adapter_id}")
    save_registry(registry_path, registry)
    return promoted
