"""Evaluate knowledge retrieval (spec §11.10) on eval/datasets/rag/v1.

Routing (deterministic, no database):

    uv run python scripts/eval_rag.py routing --split test

Retrieval: ingests the committed corpus (data/knowledge) into a temporary
PostgreSQL database (needs pgvector and permission to create databases) with
the configured embedding provider, by default the local model, then asks
every query. On dev it sweeps the similarity threshold and selects the best;
on test pass the dev threshold unchanged. Several chunking settings can be
compared in one run (each gets its own database):

    uv run python scripts/eval_rag.py retrieval --split dev --chunking 120/20/20 180/30/25 260/40/30
    uv run python scripts/eval_rag.py retrieval --split test --min-semantic 0.70

Generation: the same temporary corpus, then each query goes through the
turn's knowledge step (routing, retrieval, evidence block) and the configured
reply model with the configured companion prompt. Citation validity and
source claims without evidence are measured in code; blind packets for
groundedness judging are written next to the result. Replies are cached
(eval/results/.cache/), so a rate-limited run continues where it stopped:

    uv run python scripts/eval_rag.py generation --split test --limit 30 --pause 25

Metrics. Answerable queries: hit rate (a relevant source among the results),
source recall and precision, mean reciprocal rank, and fact hit rate (a
result contains an expected fact). Unanswerable and off-topic queries:
abstention rate (nothing returned). Routing: unnecessary retrieval rate
(lookups for messages that need none) and missed rate. Results (ids, labels,
source keys and scores only) are written to eval/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.context.builder import ContextBuilder  # noqa: E402
from app.decisions.schemas import DecisionSet  # noqa: E402
from app.embeddings.base import EmbeddingClient  # noqa: E402
from app.embeddings.factory import build_embedding_client  # noqa: E402
from app.llm.factory import build_llm_client  # noqa: E402
from app.llm.providers.registry import resolve_provider  # noqa: E402
from app.llm.schemas import (  # noqa: E402
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    TraceMetadata,
)
from app.main import COMPANION_PROMPT_ID  # noqa: E402
from app.orchestration.policy import choose_policy  # noqa: E402
from app.prompting import PromptRegistry  # noqa: E402
from app.rag.chunking import ChunkParams  # noqa: E402
from app.rag.evidence import check_citations  # noqa: E402
from app.rag.factory import retrieval_params  # noqa: E402
from app.rag.ingest import ingest, ready  # noqa: E402
from app.rag.manifest import DEFAULT_MANIFEST, load_manifest  # noqa: E402
from app.rag.retrieval import KnowledgeRetriever, RetrievalParams, merge  # noqa: E402
from app.rag.routing import route  # noqa: E402
from app.rag.service import KnowledgeService  # noqa: E402
from app.rag.store import NO_FILTER, Candidate, KnowledgeStore  # noqa: E402

DATASET_DIR = ROOT / "eval" / "datasets" / "rag" / "v1"
RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
# A reply that names a source or study as its basis. Without evidence this is
# a claim that something was looked up (spec §11.9).
SOURCE_CLAIM = re.compile(
    r"\b(?:according to|research (?:shows|suggests|says)|studies (?:show|suggest|have shown)|"
    r"a study|the (?:cdc|nih|nimh|nhs|who|mayo clinic)|guidelines? (?:say|recommend))\b",
    re.IGNORECASE,
)
NO_SOURCE = re.compile(
    r"\b(?:don't|do not|didn't) have (?:a |any )?(?:reliable |good )?(?:source|sources|reference)"
    r"|\bno (?:reliable )?(?:source|sources)\b|\bcouldn't find\b|\bnot (?:in|from) (?:my|the) "
    r"(?:library|sources)\b",
    re.IGNORECASE,
)
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(41)]  # 0.50 .. 0.90
LEXICAL_THRESHOLDS = [round(0.02 * i, 2) for i in range(1, 26)]  # 0.02 .. 0.50
POOL = 20


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}-{stamp()}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if split == "all" or r["split"] == split]


# --- routing ---------------------------------------------------------------------


async def evidence_for(
    lookups: list[dict[str, Any]], min_semantic: float | None
) -> dict[str, bool]:
    """Whether each lookup would put any passage into the prompt (by id)."""
    settings = get_settings()
    embeddings = build_embedding_client(settings)
    if embeddings is not None:
        await ready(embeddings)
    params = RetrievalParams(
        max_results=settings.knowledge_max_results,
        min_semantic=min_semantic if min_semantic is not None else settings.knowledge_min_semantic,
        min_lexical=settings.knowledge_min_lexical,
        max_per_document=settings.knowledge_max_per_document,
    )
    chunking = ChunkParams(
        max_words=settings.knowledge_chunk_max_words,
        overlap_words=settings.knowledge_chunk_overlap_words,
        min_words=settings.knowledge_chunk_min_words,
    )
    try:
        scored, _ = await candidates_for(settings.database_url, chunking, embeddings, lookups)
    finally:
        if embeddings is not None:
            await embeddings.aclose()
    return {s.query["id"]: bool(merge(s.lexical, s.semantic, params)[0]) for s in scored}


def eval_routing(args: argparse.Namespace) -> int:
    path = DATASET_DIR / "routing.jsonl"
    rows = load_jsonl(path, args.split)
    tp = fp = fn = tn = 0
    per_kind: dict[str, Counter[str]] = defaultdict(Counter)
    errors = []
    for row in rows:
        # A source follow-up needs an earlier question to look up.
        chosen = route(row["text"], previous_user_message="how much sleep do adults need?")
        want = bool(row["retrieve"])
        per_kind[row["kind"]]["n"] += 1
        per_kind[row["kind"]]["correct"] += chosen.retrieve == want
        if chosen.retrieve and want:
            tp += 1
        elif chosen.retrieve:
            fp += 1
        elif want:
            fn += 1
        else:
            tn += 1
        if chosen.retrieve != want:
            errors.append(
                {
                    "id": row["id"],
                    "want": want,
                    "kind": row["kind"],
                    "routed": chosen.kind,
                    "reason": chosen.reason,
                }
            )
    positives, negatives = tp + fn, fp + tn
    evidence: dict[str, Any] | None = None
    if args.with_retrieval:
        # What reaches the prompt: a lookup whose passages pass the threshold.
        lookups = []
        for row in rows:
            chosen = route(row["text"], previous_user_message="how much sleep do adults need?")
            if chosen.retrieve:
                lookups.append({"id": row["id"], "query": chosen.query})
        given = asyncio.run(evidence_for(lookups, args.min_semantic))
        by_kind: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            by_kind[row["kind"]]["n"] += 1
            by_kind[row["kind"]]["evidence"] += given.get(row["id"], False)
        pos_rows = [r for r in rows if r["retrieve"]]
        neg_rows = [r for r in rows if not r["retrieve"]]
        evidence = {
            "min_semantic": args.min_semantic,
            "evidence_on_negatives": round(
                sum(given.get(r["id"], False) for r in neg_rows) / len(neg_rows), 4
            )
            if neg_rows
            else None,
            "evidence_on_positives": round(
                sum(given.get(r["id"], False) for r in pos_rows) / len(pos_rows), 4
            )
            if pos_rows
            else None,
            "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
            "negatives_with_evidence": [
                {"id": r["id"], "kind": r["kind"]} for r in neg_rows if given.get(r["id"])
            ],
        }
    metrics = {
        "n": len(rows),
        "positives": positives,
        "negatives": negatives,
        "unnecessary_retrieval_rate": round(fp / negatives, 4) if negatives else None,
        "missed_rate": round(fn / positives, 4) if positives else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / positives, 4) if positives else None,
        "per_kind": {k: dict(v) for k, v in sorted(per_kind.items())},
        # Lookups run; the retrieval threshold decides what reaches the prompt.
        "lookups_on_negatives": round(fp / negatives, 4) if negatives else None,
        "evidence": evidence,
    }
    payload = {
        "eval": "rag-routing",
        "split": args.split,
        "dataset": str(path.relative_to(ROOT)),
        "dataset_sha256": file_sha256(path),
        "metrics": metrics,
        "errors": errors,
    }
    out = write_result(f"rag-routing-{args.split}", payload)
    print(json.dumps(metrics, indent=2))
    for e in errors:
        print("  miss", e)
    print(f"written: {out.relative_to(ROOT)}")
    return 0


# --- retrieval -------------------------------------------------------------------


@dataclass
class Scored:
    query: dict[str, Any]
    lexical: list[Candidate]
    semantic: list[Candidate] | None


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def score(
    items: list[Scored],
    params: RetrievalParams,
    *,
    lexical_only: bool,
    select: Callable[[Scored], list[Candidate]] | None = None,
) -> dict[str, Any]:
    """Metrics for one configuration. `select` replaces the app's merge (the
    reranker experiment)."""
    answerable = [s for s in items if s.query["type"] == "answerable"]
    negative = [s for s in items if s.query["type"] != "answerable"]
    hits = recall = precision = rr = facts = 0.0
    retrieved_answerable = 0
    per_query: list[dict[str, Any]] = []
    abstained = 0
    for s in items:
        if select is not None:
            passages = select(s)
        else:
            passages = [
                p.candidate
                for p in merge(s.lexical, None if lexical_only else s.semantic, params)[0]
            ]
        keys = [p.source_key for p in passages]
        relevant = set(s.query["relevant_sources"])
        record: dict[str, Any] = {"id": s.query["id"], "type": s.query["type"], "retrieved": keys}
        if s.query["type"] == "answerable":
            if passages:
                retrieved_answerable += 1
                precision += sum(k in relevant for k in keys) / len(keys)
            found = relevant & set(keys)
            hits += bool(found)
            recall += len(found) / len(relevant)
            first = next((i for i, k in enumerate(keys, 1) if k in relevant), None)
            rr += 1 / first if first else 0.0
            wanted = [_norm(f) for f in s.query["expected_facts"]]
            fact = any(any(f in _norm(p.text) for f in wanted) for p in passages)
            facts += fact
            record["fact_hit"] = fact
        else:
            abstained += not passages
        per_query.append(record)
    n_a, n_n = len(answerable) or 1, len(negative) or 1
    hit_rate, abstention = round(hits / n_a, 4), round(abstained / n_n, 4)
    metrics: dict[str, Any] = {
        "answerable": len(answerable),
        "negative": len(negative),
        "hit_rate": hit_rate,
        "source_recall": round(recall / n_a, 4),
        "precision": round(precision / retrieved_answerable, 4) if retrieved_answerable else None,
        "mrr": round(rr / n_a, 4),
        "fact_hit_rate": round(facts / n_a, 4),
        "abstention_rate": abstention,
        "abstention_by_type": {
            t: round(
                sum(1 for q in per_query if q["type"] == t and not q["retrieved"])
                / max(1, sum(1 for q in per_query if q["type"] == t)),
                4,
            )
            for t in ("unanswerable", "off_topic")
        },
    }
    # Balanced: finding an answer when there is one, and returning nothing when there isn't.
    metrics["balanced"] = round((hit_rate + abstention) / 2, 4)
    return {"metrics": metrics, "per_query": per_query}


def select(results: dict[float, dict[str, Any]]) -> float:
    # Best balanced score; ties go to the higher (more conservative) threshold.
    return max(results, key=lambda t: (results[t]["metrics"]["balanced"], t))


def alembic_config(url: str) -> AlembicConfig:
    config = AlembicConfig(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


async def temp_database(base_url: str) -> str:
    url = make_url(base_url)
    name = f"companion_rageval_{uuid.uuid4().hex[:10]}"
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


async def candidates_for(
    base_url: str,
    chunking: ChunkParams,
    embeddings: EmbeddingClient | None,
    queries: list[dict[str, Any]],
) -> tuple[list[Scored], dict[str, Any]]:
    url = await temp_database(base_url)
    engine = create_async_engine(url)
    try:
        store = KnowledgeStore(async_sessionmaker(engine, expire_on_commit=False))
        report = await ingest(
            load_manifest(DEFAULT_MANIFEST), store, params=chunking, embeddings=embeddings
        )
        if report.failed:
            raise SystemExit(f"ingestion failed: {[s.key for s in report.failed]}")
        model = embeddings.model_id if embeddings is not None else None
        scored: list[Scored] = []
        for query in queries:
            lexical = await store.lexical(query["query"], limit=POOL, filters=NO_FILTER)
            semantic = None
            if embeddings is not None and model is not None:
                vector = (await embeddings.embed([query["query"]], kind="query"))[0]
                semantic = await store.semantic(vector, model, limit=POOL, filters=NO_FILTER)
            scored.append(Scored(query, lexical, semantic))
        corpus = report.corpus
        info = {
            "chunking": chunking.identity,
            "chunks": corpus.chunks if corpus else None,
            "documents": corpus.documents if corpus else None,
        }
        return scored, info
    finally:
        await engine.dispose()
        await drop_database(base_url, url)


RERANK_THRESHOLDS = [round(-6.0 + 0.25 * i, 2) for i in range(57)]  # -6 .. 8 (logits)


def rerank_scores(model_name: str, scored: list[Scored]) -> dict[tuple[str, object], float]:
    """Cross-encoder score for every (query id, chunk id) candidate."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    settings = get_settings()
    model = TextCrossEncoder(model_name=model_name, cache_dir=settings.embedding_cache_dir)
    out: dict[tuple[str, object], float] = {}
    for s in scored:
        pool = {c.chunk_id: c for c in [*(s.semantic or ()), *s.lexical]}
        if not pool:
            continue
        ids = list(pool)
        texts = [f"{pool[i].heading}\n{pool[i].text}" for i in ids]
        for chunk_id, value in zip(ids, model.rerank(s.query["query"], texts), strict=True):
            out[(s.query["id"], chunk_id)] = float(value)
    return out


def reranked(
    scores: dict[tuple[str, object], float],
    params: RetrievalParams,
    *,
    threshold: float,
    semantic_gate: float | None,
) -> Callable[[Scored], list[Candidate]]:
    """Order by the reranker; gate on its score, or on similarity when
    `semantic_gate` is set; then de-duplicate and cap as the app does."""

    def select(s: Scored) -> list[Candidate]:
        pool = {c.chunk_id: c for c in [*(s.semantic or ()), *s.lexical]}
        semantic = {c.chunk_id: c.semantic for c in (s.semantic or ())}
        ranked = sorted(pool, key=lambda i: -scores.get((s.query["id"], i), -99.0))
        chosen: list[Candidate] = []
        texts: set[str] = set()
        per_doc: Counter[object] = Counter()
        for i in ranked:
            c = pool[i]
            if semantic_gate is not None:
                if (semantic.get(i) or 0.0) < semantic_gate:
                    continue
            elif scores.get((s.query["id"], i), -99.0) < threshold:
                continue
            if c.text_checksum in texts or per_doc[c.document_id] >= params.max_per_document:
                continue
            texts.add(c.text_checksum)
            per_doc[c.document_id] += 1
            chosen.append(c)
            if len(chosen) == params.max_results:
                break
        return chosen

    return select


async def eval_retrieval(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = DATASET_DIR / "retrieval.jsonl"
    queries = load_jsonl(path, args.split)
    embeddings = build_embedding_client(settings)
    if embeddings is not None:
        await ready(embeddings)
    base = RetrievalParams(
        max_results=settings.knowledge_max_results,
        min_semantic=settings.knowledge_min_semantic,
        min_lexical=settings.knowledge_min_lexical,
        max_per_document=settings.knowledge_max_per_document,
    )
    configs = args.chunking or [
        f"{settings.knowledge_chunk_max_words}/{settings.knowledge_chunk_overlap_words}/"
        f"{settings.knowledge_chunk_min_words}"
    ]
    runs: list[dict[str, Any]] = []
    try:
        for config in configs:
            max_words, overlap, min_words = (int(x) for x in config.split("/"))
            chunking = ChunkParams(max_words=max_words, overlap_words=overlap, min_words=min_words)
            scored, info = await candidates_for(
                settings.database_url, chunking, embeddings, queries
            )
            run: dict[str, Any] = {**info}
            if args.min_semantic is not None or embeddings is None:
                fixed = replace(base, min_semantic=args.min_semantic or base.min_semantic)
                run["fixed"] = {
                    "min_semantic": fixed.min_semantic,
                    **score(scored, fixed, lexical_only=embeddings is None),
                }
            if embeddings is not None and args.min_semantic is None:
                sweep = {
                    t: score(scored, replace(base, min_semantic=t), lexical_only=False)
                    for t in THRESHOLDS
                }
                best = select(sweep)
                run["sweep"] = {str(t): r["metrics"] for t, r in sweep.items()}
                run["selected_min_semantic"] = best
                run["selected"] = sweep[best]
            if args.reranker:
                scores = rerank_scores(args.reranker, scored)
                gate = {
                    t: score(
                        scored,
                        base,
                        lexical_only=False,
                        select=reranked(scores, base, threshold=t, semantic_gate=None),
                    )
                    for t in RERANK_THRESHOLDS
                }
                best_gate = select(gate)
                ordering = score(
                    scored,
                    base,
                    lexical_only=False,
                    select=reranked(
                        scores,
                        base,
                        threshold=0.0,
                        semantic_gate=args.min_semantic or base.min_semantic,
                    ),
                )
                run["reranker"] = {
                    "model": args.reranker,
                    "gate_selected": best_gate,
                    "gate_metrics": gate[best_gate]["metrics"],
                    "gate_sweep": {str(t): r["metrics"] for t, r in gate.items()},
                    "ordering_only": ordering["metrics"],
                }
                print(
                    f"   reranker {args.reranker}: gate {best_gate}",
                    json.dumps(gate[best_gate]["metrics"]),
                )
                print("   reranker ordering only:", json.dumps(ordering["metrics"]))
            lexical_values = (
                [args.min_lexical] if args.min_lexical is not None else LEXICAL_THRESHOLDS
            )
            lexical = {
                t: score(scored, replace(base, min_lexical=t), lexical_only=True)
                for t in lexical_values
            }
            best_lexical = select(lexical)
            run["lexical_fallback"] = {
                "selected_min_lexical": best_lexical,
                "metrics": lexical[best_lexical]["metrics"],
            }
            runs.append(run)
            shown = run.get("fixed") or run.get("selected")
            assert shown is not None
            print(
                f"{info['chunking']}: {info['chunks']} chunks, "
                f"selected min_semantic={run.get('selected_min_semantic')}"
            )
            print("  ", json.dumps(shown["metrics"]))
            print(
                "   lexical fallback:", best_lexical, json.dumps(run["lexical_fallback"]["metrics"])
            )
    finally:
        if embeddings is not None:
            await embeddings.aclose()
    payload = {
        "eval": "rag-retrieval",
        "split": args.split,
        "dataset": str(path.relative_to(ROOT)),
        "dataset_sha256": file_sha256(path),
        "manifest_sha256": file_sha256(DEFAULT_MANIFEST),
        "embedding_model": embeddings.model_id if embeddings is not None else None,
        "max_results": base.max_results,
        "max_per_document": base.max_per_document,
        "runs": runs,
    }
    out = write_result(f"rag-retrieval-{args.split}", payload)
    print(f"written: {out.relative_to(ROOT)}")
    return 0


async def generate(llm: Any, request: GenerationRequest) -> str:
    for attempt in range(8):
        try:
            response = await llm.generate(request)
            return str(response.content)
        except LLMError as exc:
            retryable = exc.category in (
                LLMErrorCategory.RATE_LIMITED,
                LLMErrorCategory.OVERLOADED,
                LLMErrorCategory.RESPONSE_INVALID,
                LLMErrorCategory.TIMEOUT,
            )
            if not retryable or attempt == 7:
                raise
            wait = min(120.0, exc.retry_after_seconds or 10.0 * 2**attempt)
            print(f"  {exc.category.value}, waiting {wait:.0f}s")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def pick(queries: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Answerable and unanswerable items alternated, so a limited run covers both."""
    answerable = [q for q in queries if q["type"] == "answerable"]
    other = [q for q in queries if q["type"] == "unanswerable"]
    mixed: list[dict[str, Any]] = []
    for i in range(max(len(answerable), len(other))):
        mixed += [q for q in (answerable[i : i + 1] + other[i : i + 1])]
    return mixed[:limit] if limit else mixed


async def eval_generation(args: argparse.Namespace) -> int:
    settings: Settings = get_settings()
    path = DATASET_DIR / "retrieval.jsonl"
    queries = pick(load_jsonl(path, args.split), args.limit)
    resolved = resolve_provider(settings)
    prompt = PromptRegistry().get(COMPANION_PROMPT_ID, settings.companion_prompt_version)
    builder = ContextBuilder(
        max_context_tokens=resolved.max_context_tokens,
        max_history_messages=settings.history_max_messages,
    )
    embeddings = build_embedding_client(settings)
    if embeddings is not None:
        await ready(embeddings)
    llm = build_llm_client(settings, resolved)
    chunking = ChunkParams(
        max_words=settings.knowledge_chunk_max_words,
        overlap_words=settings.knowledge_chunk_overlap_words,
        min_words=settings.knowledge_chunk_min_words,
    )
    cache_path = CACHE_DIR / f"rag-gen-{resolved.model.replace('/', '_')}.json"
    cache: dict[str, str] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    url = await temp_database(settings.database_url)
    engine = create_async_engine(url)
    rows: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    try:
        store = KnowledgeStore(async_sessionmaker(engine, expire_on_commit=False))
        await ingest(load_manifest(DEFAULT_MANIFEST), store, params=chunking, embeddings=embeddings)
        service = KnowledgeService(
            KnowledgeRetriever(
                store,
                embeddings,
                params=retrieval_params(settings),
                embedding_timeout_seconds=settings.embedding_timeout_seconds,
            ),
            timeout_seconds=30,
        )
        for query in queries:
            history = [ChatMessage(role=ChatRole.USER, content=query["query"])]
            lookup = await service.lookup(query["query"], history)
            policy = choose_policy(query["query"], DecisionSet(provider="off", results={}))
            guidance = "\n\n".join(prompt.fragment(n) for n in policy.fragments)
            context = builder.build(
                system=prompt.render(response_guidance=guidance),
                history=history,
                evidence=lookup.evidence_block(),
            )
            key = hashlib.sha256(
                json.dumps(
                    [
                        prompt.checksum,
                        resolved.model,
                        context.system,
                        [m.model_dump() for m in context.messages],
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if key not in cache:
                cache[key] = await generate(
                    llm,
                    GenerationRequest(
                        model=resolved.model,
                        system=context.system,
                        messages=context.messages,
                        max_output_tokens=resolved.max_output_tokens,
                        timeout_seconds=60,
                        params=GenerationParams(effort=resolved.effort),
                        trace=TraceMetadata(
                            trace_id=f"rag-{query['id']}",
                            purpose=GenerationPurpose.CONVERSATION,
                        ),
                    ),
                )
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                await asyncio.sleep(args.pause)
            reply = cache[key]
            check = check_citations(reply, lookup.items)
            markers = len(check.citations) + check.invalid
            relevant = set(query["relevant_sources"])
            row = {
                "id": query["id"],
                "type": query["type"],
                "route": lookup.route.kind,
                "status": lookup.status,
                "passages": [i.passage.candidate.source_key for i in lookup.items],
                "relevant_given": any(
                    i.passage.candidate.source_key in relevant for i in lookup.items
                ),
                "markers": markers,
                "invalid_markers": check.invalid,
                "cited": [c.marker for c in check.citations],
                "source_claim_without_evidence": not lookup.items
                and bool(SOURCE_CLAIM.search(reply)),
                "says_no_source": bool(NO_SOURCE.search(reply)),
                "words": len(reply.split()),
            }
            rows.append(row)
            packets.append(
                {
                    "id": query["id"],
                    "question": query["query"],
                    "passages": [
                        {"marker": i.marker, "text": i.passage.candidate.text} for i in lookup.items
                    ],
                    "reply": reply,
                }
            )
            print(
                f"{query['id']} {row['route']}/{row['status']} cited={row['cited']} "
                f"invalid={row['invalid_markers']}"
            )
    finally:
        await llm.aclose()
        if embeddings is not None:
            await embeddings.aclose()
        await engine.dispose()
        await drop_database(settings.database_url, url)
    with_evidence = [r for r in rows if r["passages"]]
    without = [r for r in rows if not r["passages"]]
    all_markers = sum(r["markers"] for r in rows)
    metrics = {
        "n": len(rows),
        "with_evidence": len(with_evidence),
        "citation_validity": (
            round(1 - sum(r["invalid_markers"] for r in rows) / all_markers, 4)
            if all_markers
            else None
        ),
        "cites_when_evidence_given": (
            round(sum(bool(r["cited"]) for r in with_evidence) / len(with_evidence), 4)
            if with_evidence
            else None
        ),
        "source_claim_without_evidence_rate": (
            round(sum(r["source_claim_without_evidence"] for r in without) / len(without), 4)
            if without
            else None
        ),
        "says_no_source_when_none": (
            round(sum(r["says_no_source"] for r in without) / len(without), 4) if without else None
        ),
        "mean_words": round(sum(r["words"] for r in rows) / len(rows), 1) if rows else None,
    }
    payload = {
        "eval": "rag-generation",
        "split": args.split,
        "dataset_sha256": file_sha256(path),
        "model": f"{resolved.preset.name}:{resolved.model}",
        "prompt": prompt.key,
        "metrics": metrics,
        "rows": rows,
    }
    out = write_result(f"rag-generation-{args.split}", payload)
    packet_path = out.with_suffix(".packets.json")
    packet_path.write_text(
        json.dumps(packets, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"written: {out.relative_to(ROOT)} and {packet_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    routing = sub.add_parser("routing")
    routing.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    routing.add_argument(
        "--with-retrieval", action="store_true", help="also check what reaches the prompt"
    )
    routing.add_argument("--min-semantic", type=float, help="threshold for --with-retrieval")
    retrieval = sub.add_parser("retrieval")
    retrieval.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    retrieval.add_argument("--chunking", nargs="*", help="max/overlap/min words, e.g. 180/30/25")
    retrieval.add_argument("--min-semantic", type=float, help="evaluate this threshold as is")
    retrieval.add_argument(
        "--min-lexical", type=float, help="evaluate this full-text fallback threshold as is"
    )
    retrieval.add_argument("--reranker", help="experiment: a fastembed cross-encoder model")
    generation = sub.add_parser("generation")
    generation.add_argument("--split", choices=["dev", "test", "all"], default="test")
    generation.add_argument("--limit", type=int, help="at most this many queries")
    generation.add_argument("--pause", type=float, default=20.0, help="seconds between calls")
    args = parser.parse_args()
    if args.mode == "routing":
        return eval_routing(args)
    if args.mode == "generation":
        return asyncio.run(eval_generation(args))
    return asyncio.run(eval_retrieval(args))


if __name__ == "__main__":
    raise SystemExit(main())
