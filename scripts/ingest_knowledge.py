"""Ingest the reviewed knowledge corpus into PostgreSQL (docs/rag.md).

Reads data/knowledge/manifest.toml and the snapshots written by
scripts/fetch_knowledge.py; nothing is downloaded. Safe to re-run: unchanged
sources are left as they are, changed ones get a new version.

Usage:
  uv run python scripts/ingest_knowledge.py          # DATABASE_URL and EMBEDDING_* from .env
  uv run python scripts/ingest_knowledge.py --prune  # also retire sources not in the manifest
  uv run python scripts/ingest_knowledge.py --only cdc-sleep-about --report out.json
Exit status 1 when a source failed or a probe query missed its source.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.embeddings.base import EmbeddingError
from app.embeddings.factory import build_embedding_client
from app.rag.factory import build_retriever, chunk_params
from app.rag.ingest import IngestReport, ingest, ready
from app.rag.manifest import DEFAULT_MANIFEST, ManifestError, load_manifest
from app.rag.store import KnowledgeStore


def show(report: IngestReport) -> None:
    print(f"chunking {report.chunking}; embeddings {report.embedding_model or 'none'}")
    for source in report.sources:
        detail = source.failure or (
            f"v{source.version}, {source.chunks} chunks" if source.chunks else f"v{source.version}"
        )
        if source.duplicate_of:
            detail = f"same text as {source.duplicate_of}"
        print(f"  {source.outcome:<17}{source.key}: {detail}")
    print("counts:", report.counts)
    if report.retired:
        print("retired:", ", ".join(report.retired))
    print(
        f"embedded {report.embedded} chunks"
        + (f"; embedding stopped: {report.embedding_failure}" if report.embedding_failure else "")
    )
    if report.probes:
        found = sum(p.found for p in report.probes)
        print(f"probes: {found}/{len(report.probes)} found their source")
        for probe in report.probes_failed:
            print(f"  missed {probe.key}: {probe.query!r}")
    if report.corpus is not None:
        c = report.corpus
        print(
            f"corpus: {c.documents} documents ({c.retired_documents} retired versions), "
            f"{c.chunks} chunks, embedded {c.embedded}"
        )
    print(f"{report.seconds}s")


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        for problem in exc.problems:
            print(f"manifest: {problem}")
        return 2
    embeddings = None if args.no_embeddings else build_embedding_client(settings)
    engine = create_engine(settings)
    try:
        if embeddings is not None:
            try:
                await ready(embeddings)
            except EmbeddingError as exc:
                print(f"embeddings unavailable ({exc.message}); ingesting without vectors")
                embeddings = None
        sessions = create_session_factory(engine)
        report = await ingest(
            manifest,
            KnowledgeStore(sessions),
            params=chunk_params(settings),
            embeddings=embeddings,
            retriever=build_retriever(settings, sessions, embeddings),
            prune=args.prune,
            only=set(args.only) if args.only else None,
        )
    finally:
        if embeddings is not None:
            await embeddings.aclose()
        await engine.dispose()
    show(report)
    if args.report:
        args.report.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    return 1 if report.failed or report.probes_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the knowledge corpus.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", nargs="*", help="source keys")
    parser.add_argument("--prune", action="store_true", help="retire sources not in the manifest")
    parser.add_argument("--no-embeddings", action="store_true", help="full-text search only")
    parser.add_argument("--report", type=Path, help="write the report as JSON")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
