"""Evaluate memory retrieval and extraction (spec §10.8) on eval/datasets/memory/v1.

Retrieval runs the real store against a temporary PostgreSQL database (it
needs pgvector and permission to create databases) with the configured
embedding provider, by default the local model:

    uv run python scripts/eval_memory.py retrieval --split dev
    uv run python scripts/eval_memory.py retrieval --split test --min-semantic 0.62

On dev it sweeps the semantic threshold and selects the one with the best F1;
on test pass the dev threshold unchanged.

Extraction calls the configured generation model (GENERATION_PROVIDER, or
MEMORY_EXTRACTION_MODEL) with the memory-extraction prompt, then applies the
eligibility policy in code, exactly as after a live turn:

    uv run python scripts/eval_memory.py extraction --split test

Answers are cached (eval/results/.cache/), so a run stopped by a rate limit
continues where it left off. Results are written to eval/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.config import Settings  # noqa: E402
from app.db import repositories as repo  # noqa: E402
from app.db.models import MemoryRecord  # noqa: E402
from app.embeddings.base import EmbeddingClient  # noqa: E402
from app.embeddings.factory import build_embedding_client  # noqa: E402
from app.evaluation.memory import (  # noqa: E402
    MATCH_THRESHOLD,
    ExtractionItem,
    ItemScore,
    Scenario,
    extraction_metrics,
    load_extraction,
    load_scenarios,
    retrieval_metrics,
    score_item,
    select_threshold,
)
from app.llm.factory import build_llm_client  # noqa: E402
from app.llm.providers.registry import resolve_provider  # noqa: E402
from app.main import MEMORY_PROMPT_ID, build_memory_extractor  # noqa: E402
from app.memory.policy import MemorySettings, decide  # noqa: E402
from app.memory.schemas import (  # noqa: E402
    MEMORY_SCHEMA_VERSION,
    ExtractedCandidate,
    MemoryStatus,
)
from app.memory.store import MemoryStore, RetrievalParams, select_hits  # noqa: E402
from app.prompting import PromptRegistry  # noqa: E402

DATASET_DIR = ROOT / "eval" / "datasets" / "memory" / "v1"
RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
THRESHOLDS = [round(0.30 + 0.01 * i, 2) for i in range(61)]  # 0.30 .. 0.90


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}-{stamp()}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


async def ready_embeddings(settings: Settings) -> EmbeddingClient | None:
    client = build_embedding_client(settings)
    if client is None:
        return None
    client.start()
    for _ in range(600):  # the first run downloads the local model
        try:
            await client.embed(["warm up"], kind="query")
            return client
        except Exception:
            await asyncio.sleep(0.2)
    raise SystemExit("The embedding model did not become ready.")


# --- retrieval -------------------------------------------------------------------


def alembic_config(url: str) -> AlembicConfig:
    config = AlembicConfig(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


async def temp_database(base_url: str) -> str:
    url = make_url(base_url)
    name = f"companion_memeval_{uuid.uuid4().hex[:10]}"
    admin = create_async_engine(url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await admin.dispose()
    temp = url.set(database=name).render_as_string(hide_password=False)
    await asyncio.to_thread(command.upgrade, alembic_config(temp), "head")
    return temp


async def drop_database(base_url: str, temp_url: str) -> None:
    admin = create_async_engine(make_url(base_url), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(
            text(f'DROP DATABASE IF EXISTS "{make_url(temp_url).database}" WITH (FORCE)')
        )
    await admin.dispose()


async def score_scenarios(
    settings: Settings, scenarios: list[Scenario], base_url: str, exact: RetrievalParams | None
) -> dict[str, Any]:
    """Store each scenario's memories for its own user, then ask the store for
    every candidate with its scores (so thresholds can be swept), the
    full-text-only fallback, and, with `exact`, the configured retrieval."""
    embeddings = await ready_embeddings(settings)
    model = embeddings.model_id if embeddings else None
    url = await temp_database(base_url)
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = MemoryStore(sessions)
    now = datetime.now(UTC)
    candidates: dict[str, list[Any]] = {}
    lexical: dict[str, list[str]] = {}
    exact_keys: dict[str, list[str]] = {}
    try:
        for scenario in scenarios:
            async with sessions() as session, session.begin():
                user_id = await repo.get_or_create_user(session, f"eval:{scenario.id}")
            contents = [m.content for m in scenario.memories]
            vectors = await embeddings.embed(contents, kind="document") if embeddings else None
            keys: dict[uuid.UUID, str] = {}
            async with sessions() as session, session.begin():
                for i, m in enumerate(scenario.memories):
                    memory_id = uuid.uuid4()
                    keys[memory_id] = m.key
                    created = now - timedelta(days=m.days_ago)
                    session.add(
                        MemoryRecord(
                            id=memory_id,
                            user_id=user_id,
                            memory_type=m.memory_type,
                            content=m.content,
                            status=MemoryStatus.SUPERSEDED
                            if m.status == "superseded"
                            else MemoryStatus.ACTIVE,
                            importance=2,
                            provenance={"eval": scenario.id},
                            expires_at=now - timedelta(days=1) if m.status == "expired" else None,
                            embedding=vectors[i] if vectors else None,
                            embedding_model=model if vectors else None,
                            schema_version=MEMORY_SCHEMA_VERSION,
                            created_at=created,
                            updated_at=created,
                        )
                    )
            query = (
                (await embeddings.embed([scenario.message], kind="query"))[0]
                if embeddings
                else None
            )
            everything = await store.retrieve(
                user_id,
                query=scenario.message,
                query_embedding=query,
                embedding_model=model,
                now=now,
                params=RetrievalParams(limit=10_000, min_semantic=-2.0, lexical_only_ok=True),
            )
            candidates[scenario.id] = [(r, keys[r.memory.id]) for r in everything]
            fallback = await store.retrieve(
                user_id,
                query=scenario.message,
                query_embedding=None,
                embedding_model=None,
                now=now,
                params=RetrievalParams(limit=settings.memory_max_retrieved),
            )
            lexical[scenario.id] = [keys[r.memory.id] for r in fallback]
            if exact is not None:
                found = await store.retrieve(
                    user_id,
                    query=scenario.message,
                    query_embedding=query,
                    embedding_model=model,
                    now=now,
                    params=exact,
                )
                exact_keys[scenario.id] = [keys[r.memory.id] for r in found]
    finally:
        await engine.dispose()
        await drop_database(base_url, url)
        if embeddings is not None:
            await embeddings.aclose()
    return {"model": model, "candidates": candidates, "lexical": lexical, "exact": exact_keys}


def keys_at(candidates: dict[str, list[Any]], params: RetrievalParams) -> dict[str, list[str]]:
    result = {}
    for scenario_id, rows in candidates.items():
        key_of = {r.memory.id: key for r, key in rows}
        hits = select_hits([r for r, _ in rows], params)
        result[scenario_id] = [key_of[r.memory.id] for r in hits]
    return result


def run_retrieval(args: argparse.Namespace) -> int:
    settings = Settings()
    dataset = DATASET_DIR / f"retrieval_{args.split}.jsonl"
    scenarios = load_scenarios(dataset)
    limit = settings.memory_max_retrieved
    exact = (
        RetrievalParams(limit=limit, min_semantic=args.min_semantic, lexical_only_ok=False)
        if args.min_semantic is not None
        else None
    )
    base_url = args.database_url or settings.database_url
    scored = asyncio.run(score_scenarios(settings, scenarios, base_url, exact))
    candidates = scored["candidates"]
    sweep: dict[str, list[dict[str, Any]]] = {}
    for mode in ("semantic", "hybrid_or"):
        sweep[mode] = []
        for t in THRESHOLDS:
            params = RetrievalParams(limit, t, lexical_only_ok=(mode == "hybrid_or"))
            metrics = retrieval_metrics(scenarios, keys_at(candidates, params))
            metrics.pop("by_tag")  # kept for the selected threshold only
            sweep[mode].append({"min_semantic": t, "metrics": metrics})
    ranking_only = retrieval_metrics(scenarios, keys_at(candidates, RetrievalParams(limit, -2.0)))
    payload: dict[str, Any] = {
        "kind": "memory-retrieval",
        "split": args.split,
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": file_sha256(dataset),
        "embedding_model": scored["model"],
        "limit": limit,
        "lexical_fallback": retrieval_metrics(scenarios, scored["lexical"]),
        "top_k_without_threshold": ranking_only,
        "sweep": sweep,
        "scores": {
            sid: [
                {"key": key, "semantic": r.semantic, "lexical": round(r.lexical, 4)}
                for r, key in rows
            ]
            for sid, rows in candidates.items()
        },
    }
    if exact is not None:
        payload["min_semantic"] = args.min_semantic
        payload["metrics"] = retrieval_metrics(scenarios, scored["exact"])
        payload["retrieved"] = scored["exact"]
        # The sweep must reproduce the store's own selection.
        assert keys_at(candidates, exact) == scored["exact"], "sweep diverges from the store"
        print(f"Test at min_semantic={args.min_semantic}: {summary(payload['metrics'])}")
    else:
        chosen = select_threshold(sweep["semantic"])
        payload["selected_min_semantic"] = chosen
        if chosen is not None:
            picked = keys_at(candidates, RetrievalParams(limit, chosen))
            payload["metrics"] = retrieval_metrics(scenarios, picked)
            payload["retrieved"] = picked
            print(f"Dev: best min_semantic={chosen}: {summary(payload['metrics'])}")
    print(f"Top {limit}, no threshold: {summary(ranking_only)}")
    print(f"Full-text fallback: {summary(payload['lexical_fallback'])}")
    out = write_result(f"memory-retrieval-{args.split}", payload)
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


def summary(metrics: dict[str, Any]) -> str:
    empty = metrics["nothing_relevant"]
    return (
        f"precision {metrics['precision']} recall {metrics['recall']} f1 {metrics['f1']}; "
        f"nothing returned for {empty['returned_nothing']}/{empty['scenarios']} "
        f"no-relevant scenarios; inactive returned {len(metrics['inactive_returned'])}"
    )


# --- extraction --------------------------------------------------------------------


def read_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")


async def propose_all(
    settings: Settings, items: list[ExtractionItem], cache_path: Path, pause: float
) -> tuple[dict[str, Any], str, str]:
    resolved = resolve_provider(settings)
    llm = build_llm_client(settings, resolved)
    prompt = PromptRegistry().get(MEMORY_PROMPT_ID, settings.memory_prompt_version)
    extractor = build_memory_extractor(settings, resolved, llm, prompt)
    assert extractor is not None, "MEMORY_EXTRACTION_ENABLED is false"
    cache: dict[str, Any] = await asyncio.to_thread(read_cache, cache_path)
    print(f"Extraction model {resolved.preset.name}:{extractor.model}, prompt {prompt.key}")
    try:
        for n, item in enumerate(items, 1):
            if item.id in cache and cache[item.id]["failure"] is None:
                continue
            for attempt in range(8):
                proposal = await extractor.propose(
                    [], item.history, trace_id=f"eval-{item.id}", conversation_id=uuid.uuid4()
                )
                retryable = proposal.failure in ("llm:rate_limited", "llm:overloaded", "timeout")
                if not retryable:
                    break
                wait = min(120, 10 * 2**attempt)
                print(f"  {item.id}: {proposal.failure}, waiting {wait}s")
                await asyncio.sleep(wait)
            earlier = cache.get(item.id, {})
            cache[item.id] = {
                # Failures before this answer (a rerun re-asks failed items);
                # live turns do not retry, so these count as failures.
                "earlier_failures": [
                    *earlier.get("earlier_failures", []),
                    *([earlier["failure"]] if earlier.get("failure") else []),
                ],
                "failure": proposal.failure,
                "model_id": proposal.model_id,
                "structured_mode": proposal.structured_mode,
                "candidates": [c.model_dump() for c in proposal.output.candidates]
                if proposal.output
                else [],
            }
            await asyncio.to_thread(write_cache, cache_path, cache)
            print(f"[{n}/{len(items)}] {item.id}: {proposal.failure or 'ok'}")
            if proposal.failure == "llm:quota_exhausted":
                print("Daily quota exhausted; rerun later to continue from the cache.")
                break
            await asyncio.sleep(pause)
    finally:
        await llm.aclose()
    return cache, f"{resolved.preset.name}:{extractor.model}", prompt.key


async def bge_similarity(settings: Settings, texts: set[str]) -> Any:
    """Cosine similarity between bge-small-en-v1.5 vectors (always the local
    model, whatever EMBEDDING_PROVIDER is, so scores are comparable)."""
    local = settings.model_copy(update={"embedding_provider": "local", "embedding_model": None})
    client = await ready_embeddings(local)
    assert client is not None
    ordered = sorted(texts)
    vectors = dict(zip(ordered, await client.embed(ordered, kind="document"), strict=True))
    await client.aclose()

    def similarity(a: str, b: str) -> float:
        return sum(x * y for x, y in zip(vectors[a], vectors[b], strict=True))

    return similarity


def run_extraction(args: argparse.Namespace) -> int:
    settings = Settings()
    dataset = DATASET_DIR / f"extraction_{args.split}.jsonl"
    items = load_extraction(dataset)
    if args.limit:
        items = items[: args.limit]
    model_key = (
        settings.memory_extraction_model or settings.generation_model or "default"
    ).replace("/", "_")
    cache_path = CACHE_DIR / (
        f"extraction-{settings.generation_provider}-{model_key}-"
        f"{settings.memory_prompt_version}-{args.split}.json"
    )
    cache, model, prompt_key = asyncio.run(propose_all(settings, items, cache_path, args.pause))
    answered = [i for i in items if i.id in cache]
    now = datetime.now(UTC)
    kept: dict[str, list[str]] = {}
    raw: dict[str, list[str]] = {}
    reasons: dict[str, list[str]] = {}
    for item in answered:
        candidates = [ExtractedCandidate.model_validate(c) for c in cache[item.id]["candidates"]]
        decisions = decide(
            candidates,
            latest_message=item.message,
            shown=[],
            settings=MemorySettings(),
            now=now,
            retention_days=settings.memory_retention_days,
            max_per_turn=settings.memory_max_per_turn,
        )
        kept[item.id] = [d.content for d in decisions if d.action != "reject"]
        raw[item.id] = [c.content for c in candidates]
        reasons[item.id] = [f"{d.action}:{d.reason}" for d in decisions]
    texts = {t for i in answered for t in (*kept[i.id], *raw[i.id], *i.minority)}
    texts |= {p for i in answered for g in i.expected for p in g.phrasings}
    similarity = asyncio.run(bge_similarity(settings, texts or {""}))

    def scored(predictions: dict[str, list[str]]) -> list[ItemScore]:
        return [
            score_item(i, predictions[i.id], similarity, failure=cache[i.id]["failure"])
            for i in answered
        ]

    stored, proposed = scored(kept), scored(raw)
    payload = {
        "kind": "memory-extraction",
        "split": args.split,
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": file_sha256(dataset),
        "model": model,
        "prompt": prompt_key,
        "match_threshold": MATCH_THRESHOLD,
        "items_answered": len(answered),
        # Items whose first request failed (no memory would be stored live).
        "first_attempt_failures": sorted(
            i.id for i in answered if cache[i.id]["failure"] or cache[i.id].get("earlier_failures")
        ),
        "items_requested": len(items),
        # What the system stores (model candidates after the code-side policy).
        "stored": extraction_metrics(answered, stored),
        # The model's raw candidates, before the policy.
        "proposed": extraction_metrics(answered, proposed),
        "per_item": [
            {
                "id": i.id,
                "message": i.message,
                "gold": [g.content for g in i.expected],
                "proposed": raw[i.id],
                "decisions": reasons[i.id],
                "stored": s.predicted,
                "unsupported": s.unsupported,
                "missed": s.missed,
                "failure": s.failure,
            }
            for i, s in zip(answered, stored, strict=True)
        ],
    }
    for name in ("stored", "proposed"):
        m = payload[name]
        print(
            f"{name}: precision {m['memories']['precision']} recall {m['memories']['recall']} "
            f"f1 {m['memories']['f1']}; stored something for "
            f"{m['nothing_to_store']['stored_something']}/{m['nothing_to_store']['items']} "
            f"items with nothing to store; failures {m['failures']}"
        )
    out = write_result(f"memory-extraction-{args.split}", payload)
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    retrieval = commands.add_parser("retrieval", help="retrieval precision/recall (local)")
    retrieval.add_argument("--split", choices=("dev", "test"), default="dev")
    retrieval.add_argument("--min-semantic", type=float, help="threshold chosen on dev")
    retrieval.add_argument("--database-url", help="default: DATABASE_URL")
    retrieval.set_defaults(func=run_retrieval)
    extraction = commands.add_parser("extraction", help="extraction precision/recall (live LLM)")
    extraction.add_argument("--split", choices=("dev", "test"), default="dev")
    extraction.add_argument("--limit", type=int, help="first N items only")
    extraction.add_argument("--pause", type=float, default=2.0, help="seconds between calls")
    extraction.set_defaults(func=run_extraction)
    args = parser.parse_args()
    started = time.monotonic()
    code: int = args.func(args)
    print(f"Done in {time.monotonic() - started:.0f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
