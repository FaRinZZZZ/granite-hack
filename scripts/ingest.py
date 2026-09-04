"""Build the FAISS index from everything in data/.

    python -m scripts.ingest
    python -m scripts.ingest --data-dir data --index-dir storage
"""
import argparse
import logging
from pathlib import Path

from app import config
from app.rag import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.rag.data_dir)
    parser.add_argument("--index-dir", type=Path, default=config.rag.index_dir)
    args = parser.parse_args()

    store = build_index(args.data_dir, args.index_dir)
    print(f"\nindexed {len(store.chunks)} chunks -> {args.index_dir}")

    sources: dict[str, int] = {}
    for chunk in store.chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count} chunks")


if __name__ == "__main__":
    main()
