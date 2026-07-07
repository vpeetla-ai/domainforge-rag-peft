from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_manifest(
    data_root: Path,
    sop_map_path: Path,
    split_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corpus_files = sorted((data_root / "corpus" / "sop_documents").glob("*.md"))
    manifests: dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sop_map_sha256": sha256_file(sop_map_path),
        "corpus_files": [
            {"path": str(p.relative_to(data_root)), "sha256": sha256_file(p)} for p in corpus_files
        ],
    }
    for name in ("train", "val", "test"):
        split_path = data_root / "sft_pairs" / f"{name}.jsonl"
        if split_path.exists():
            manifests[f"sft_{name}"] = {
                "path": str(split_path.relative_to(data_root)),
                "sha256": sha256_file(split_path),
                "lines": sum(1 for _ in split_path.open(encoding="utf-8")),
            }
    if split_stats:
        manifests["split_stats"] = split_stats
    return manifests


def write_manifest(data_root: Path, sop_map_path: Path, split_stats: dict[str, Any] | None = None) -> Path:
    manifest = build_dataset_manifest(data_root, sop_map_path, split_stats)
    out = data_root / "manifests" / "dataset_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
