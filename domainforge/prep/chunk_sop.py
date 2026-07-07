from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_sop_map(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sop_stem(filename: str) -> str:
    return Path(filename).stem


def section_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "section"


def chunk_id_for(source_file: str, section_title: str) -> str:
    return f"{sop_stem(source_file)}__{section_slug(section_title)}"


def cite_ids_for_intent(sop_map: dict[str, Any], intent: str) -> list[str]:
    """Return expected chunk_id prefixes for an intent from the SOP manifest."""
    ids: list[str] = []
    for doc in sop_map.get("documents", []):
        if intent in doc.get("intent_tags", []):
            source = doc["source_file"]
            for section in doc.get("sections", ["Scope"]):
                ids.append(chunk_id_for(source, section))
    if ids:
        return ids

    fallback = sop_map.get("intent_gaps", {}).get("fallback_sop", {})
    fallback_file = fallback.get(intent)
    if fallback_file:
        for doc in sop_map.get("documents", []):
            if doc["source_file"] == fallback_file:
                return [chunk_id_for(fallback_file, doc.get("sections", ["Scope"])[0])]
    return []


def chunk_markdown_sop(
    source_file: Path,
    doc_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split an SOP markdown file into section-level chunks with metadata."""
    text = source_file.read_text(encoding="utf-8")
    chunks: list[dict[str, Any]] = []
    current_title = "Preamble"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        body = "\n".join(current_lines).strip()
        if not body:
            return
        chunks.append(
            {
                "chunk_id": chunk_id_for(source_file.name, current_title),
                "source_sop": source_file.name,
                "section_title": current_title,
                "text": body,
                "intent_tags": doc_meta.get("intent_tags", []),
                "category": doc_meta.get("category", ""),
            }
        )
        current_lines = []

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current_title = line[3:].strip()
        else:
            current_lines.append(line)
    flush()
    return chunks


def chunk_all_sops(corpus_dir: Path, sop_map_path: Path) -> list[dict[str, Any]]:
    sop_map = load_sop_map(sop_map_path)
    meta_by_file = {d["source_file"]: d for d in sop_map.get("documents", [])}
    all_chunks: list[dict[str, Any]] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        meta = meta_by_file.get(md_file.name, {"intent_tags": [], "category": ""})
        all_chunks.extend(chunk_markdown_sop(md_file, meta))
    return all_chunks
