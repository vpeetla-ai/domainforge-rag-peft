from __future__ import annotations

import argparse
import json
from pathlib import Path

from domainforge.config import get_settings
from domainforge.rag.factory import index_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma S1 index from SOP chunks")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--sop-map", type=Path, default=None)
    parser.add_argument("--chroma-path", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.corpus_dir:
        settings.corpus_dir = args.corpus_dir
    if args.sop_map:
        settings.manifest_path = args.sop_map
    if args.chroma_path:
        settings.chroma_path = args.chroma_path

    result = index_corpus(settings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
