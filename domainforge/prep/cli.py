from __future__ import annotations

import argparse
import json
from pathlib import Path

from domainforge.config import get_settings
from domainforge.prep.bitext import build_sft_splits
from domainforge.prep.chunk_sop import chunk_all_sops
from domainforge.prep.manifests import write_manifest


def cmd_chunk_sops(corpus_dir: Path, sop_map_path: Path, out_path: Path) -> None:
    chunks = chunk_all_sops(corpus_dir, sop_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {out_path}")


def cmd_fetch_bitext(out_dir: Path, sop_map_path: Path, max_rows: int | None) -> None:
    stats = build_sft_splits(out_dir, sop_map_path, max_rows=max_rows)
    print(json.dumps(stats, indent=2))


def cmd_build_manifests(data_root: Path, sop_map_path: Path) -> None:
    out = write_manifest(data_root, sop_map_path)
    print(f"Wrote manifest to {out}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="DomainForge data preparation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chunk = sub.add_parser("chunk-sops", help="Chunk SOP markdown into retrieval units")
    p_chunk.add_argument("--corpus-dir", type=Path, default=settings.corpus_dir)
    p_chunk.add_argument("--sop-map", type=Path, default=settings.manifest_path)
    p_chunk.add_argument("--out", type=Path, default=Path("data/corpus/sop_chunks.json"))

    p_bitext = sub.add_parser("fetch-bitext", help="Download Bitext and build SFT splits")
    p_bitext.add_argument("--out-dir", type=Path, default=Path("data/sft_pairs"))
    p_bitext.add_argument("--sop-map", type=Path, default=settings.manifest_path)
    p_bitext.add_argument("--max-rows", type=int, default=None)

    p_manifest = sub.add_parser("build-manifests", help="Write dataset_v1.json manifest")
    p_manifest.add_argument("--data-root", type=Path, default=settings.data_root)
    p_manifest.add_argument("--sop-map", type=Path, default=settings.manifest_path)

    p_prefs = sub.add_parser("build-preferences", help="Build DPO preference pairs from golden eval")
    p_prefs.add_argument("--golden", type=Path, default=Path("data/eval_golden/sample.jsonl"))
    p_prefs.add_argument("--sop-map", type=Path, default=settings.manifest_path)
    p_prefs.add_argument("--corpus-dir", type=Path, default=settings.corpus_dir)
    p_prefs.add_argument("--out-dir", type=Path, default=Path("data/preferences"))

    args = parser.parse_args()
    if args.command == "chunk-sops":
        cmd_chunk_sops(args.corpus_dir, args.sop_map, args.out)
    elif args.command == "fetch-bitext":
        cmd_fetch_bitext(args.out_dir, args.sop_map, args.max_rows)
    elif args.command == "build-manifests":
        cmd_build_manifests(args.data_root, args.sop_map)
    elif args.command == "build-preferences":
        from domainforge.prep.preferences import build_preference_splits

        stats = build_preference_splits(
            args.golden,
            args.sop_map,
            args.corpus_dir,
            args.out_dir,
        )
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
