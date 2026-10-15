"""Run the golden scenarios end to end (spec §16.3; docs/evaluation.md).

Each scenario runs through the real application: HTTP API, a temporary
PostgreSQL database with the knowledge corpus ingested, the orchestrator
and every layer.

    uv run python scripts/eval_golden.py --mode deterministic          # no key, CI
    uv run python scripts/eval_golden.py --mode live --split dev --limit 10 --pause 20

- **Deterministic:** the development stub model, no safety classifier
  (degraded mode), no decision model. Pipeline checks only.
- **Live:** the configured provider and settings. Live replies are cached
  per scenario (eval/results/.cache/), so a run stopped by a rate limit
  continues where it stopped.

Results go to eval/results/golden-<mode>-<split>-<time>.json, and the
replies (synthetic conversations) to a `.replies.json` file next to it.
`--record-db` also records the run in `evaluation_runs` of DATABASE_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.config import Settings  # noqa: E402
from app.embeddings.factory import build_embedding_client  # noqa: E402
from app.evaluation.golden import (  # noqa: E402
    Mode,
    Observed,
    Scenario,
    ScenarioOutcome,
    check_step,
    load_scenarios,
    summarize,
)
from app.evaluation.metadata import experiment_metadata  # noqa: E402
from app.evaluation.runs import ItemResult, RunRecorder  # noqa: E402
from app.llm.base import LLMClient  # noqa: E402
from app.llm.factory import build_llm_client  # noqa: E402
from app.llm.providers.registry import resolve_provider  # noqa: E402
from app.llm.schemas import (  # noqa: E402
    GenerationPurpose,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    LLMErrorCategory,
)
from app.main import create_app  # noqa: E402
from app.rag.factory import chunk_params  # noqa: E402
from app.rag.ingest import ingest, ready  # noqa: E402
from app.rag.manifest import load_manifest  # noqa: E402
from app.rag.store import KnowledgeStore  # noqa: E402
from app.safety.factory import classifier_provider, resolve_safety_llm  # noqa: E402

DATASET = ROOT / "eval" / "datasets" / "golden" / "v1" / "scenarios.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
USERS = {"alice": "token-alice", "bob": "token-bob"}
LIMITATIONS = [
    "synthetic scenarios written and labelled by AI agents, not human-reviewed",
    "reply checks are surface proxies (questions, lists, stock phrases, mentions)",
]


class FaultInjector:
    """Wraps an LLM client; `arm(purpose)` makes the next request with that
    purpose fail as a provider error (failure-matrix scenarios)."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self._armed: set[GenerationPurpose] = set()

    @property
    def provider(self) -> str:
        return self.inner.provider

    def arm(self, purpose: GenerationPurpose) -> None:
        self._armed.add(purpose)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.trace.purpose in self._armed:
            self._armed.discard(request.trace.purpose)
            raise LLMError(LLMErrorCategory.SERVER_ERROR, "injected failure")
        return await self.inner.generate(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


class CachingLLM:
    """Live mode: every response is cached by its request, so a rerun (or a
    run resumed after a rate limit) makes no new calls for steps already
    done. Transient provider errors are retried with backoff."""

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner = inner
        self.path = path
        self.cache: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        self.calls = 0

    @property
    def provider(self) -> str:
        return self.inner.provider

    @staticmethod
    def key(request: GenerationRequest) -> str:
        body = request.model_dump(mode="json", exclude={"trace", "timeout_seconds"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        key = self.key(request)
        if key in self.cache:
            return GenerationResponse.model_validate(self.cache[key])
        for attempt in range(6):
            try:
                response = await self.inner.generate(request)
                break
            except LLMError as exc:
                retryable = exc.category in (
                    LLMErrorCategory.RATE_LIMITED,
                    LLMErrorCategory.OVERLOADED,
                    LLMErrorCategory.TIMEOUT,
                )
                if not retryable or attempt == 5:
                    raise
                await asyncio.sleep(min(90.0, exc.retry_after_seconds or 10.0 * 2**attempt))
        self.calls += 1
        self.cache[key] = response.model_dump(mode="json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache), encoding="utf-8")
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


class KnowledgeFault:
    """Makes the next knowledge search fail as an unreachable store."""

    def __init__(self) -> None:
        self.armed = False
        self._original = KnowledgeStore.lexical

    def install(self) -> None:
        fault = self

        async def lexical(store: KnowledgeStore, *args: Any, **kwargs: Any) -> Any:
            if fault.armed:
                fault.armed = False
                raise OSError("injected failure")
            return await fault._original(store, *args, **kwargs)

        KnowledgeStore.lexical = lexical  # type: ignore[method-assign,assignment]

    def uninstall(self) -> None:
        KnowledgeStore.lexical = self._original  # type: ignore[method-assign]


def alembic_config(url: str) -> AlembicConfig:
    config = AlembicConfig(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


@asynccontextmanager
async def temp_database(base_url: str) -> AsyncIterator[str]:
    url = make_url(base_url)
    name = f"companion_golden_{uuid.uuid4().hex[:10]}"
    admin = create_async_engine(url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    temp = url.set(database=name).render_as_string(hide_password=False)
    try:
        await asyncio.to_thread(command.upgrade, alembic_config(temp), "head")
        yield temp
    finally:
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await admin.dispose()


def settings_for(mode: Mode, base: Settings, database_url: str) -> Settings:
    common: dict[str, Any] = {
        "database_url": database_url,
        "dev_auth_tokens": ",".join(f"{u}:{t}" for u, t in USERS.items()),
        "app_env": "development",
        "user_messages_per_minute": 0,
        "user_messages_per_day": 0,
    }
    if mode == "deterministic":
        common.update(
            generation_provider="fake",
            safety_classifier_provider="off",
            decision_provider="off",
        )
    # Validated again (model_copy would not turn strings into secrets).
    values = {**base.model_dump(), **common}
    return Settings(_env_file=None, **values)


async def run_scenario(
    scenario: Scenario,
    mode: Mode,
    client: httpx.AsyncClient,
    app: Any,
    llm: FaultInjector,
    safety_llm: FaultInjector | None,
    knowledge_fault: KnowledgeFault,
    pause: float,
) -> tuple[ScenarioOutcome, list[dict[str, Any]]]:
    alice = {"Authorization": f"Bearer {USERS['alice']}"}
    bob = {"Authorization": f"Bearer {USERS['bob']}"}
    checks: list[tuple[int, Any]] = []
    transcript: list[dict[str, Any]] = []

    async def new_conversation() -> str:
        response = await client.post("/v1/conversations", headers=alice)
        response.raise_for_status()
        return str(response.json()["conversation_id"])

    conversation = await new_conversation()
    for index, step in enumerate(scenario.steps):
        if step.action == "new_conversation":
            await app.state.background.drain()  # memories of the last turn are stored
            conversation = await new_conversation()
            continue
        if step.action == "fail_next":
            component = step.component
            if component == "llm":
                llm.arm(GenerationPurpose.CONVERSATION)
            elif component == "state":
                llm.arm(GenerationPurpose.STATE_EXTRACTION)
            elif component == "decisions":
                llm.arm(GenerationPurpose.DECISION)
            elif component == "safety_classifier" and safety_llm is not None:
                safety_llm.arm(GenerationPurpose.SAFETY_CLASSIFICATION)
            elif component == "knowledge":
                knowledge_fault.armed = True
            continue
        if step.action == "delete_memories":
            await app.state.background.drain()
            deleted = await client.delete("/v1/users/me/memories", headers=alice)
            listed = await client.get("/v1/users/me/memories?status=all", headers=alice)
            ok = deleted.status_code == 200 and listed.json().get("memories") == []
            checks += [
                (index, c) for c in check_step(step.expect, Observed(200, probe_ok=ok), mode)
            ]
            continue
        if step.action == "privacy_probe":
            await app.state.background.drain()
            own = await client.get("/v1/users/me/memories?status=all", headers=alice)
            alice_ids = {m["memory_id"] for m in own.json().get("memories", [])}
            responses = [
                await client.get(f"/v1/conversations/{conversation}", headers=bob),
                await client.get(f"/v1/conversations/{conversation}/messages", headers=bob),
                await client.post(
                    f"/v1/conversations/{conversation}/messages",
                    json={"content": "hello"},
                    headers=bob,
                ),
                await client.delete(f"/v1/conversations/{conversation}", headers=bob),
            ]
            theirs = await client.get("/v1/users/me/memories?status=all", headers=bob)
            bob_ids = {m["memory_id"] for m in theirs.json().get("memories", [])}
            foreign = [
                await client.get(f"/v1/users/me/memories/{memory_id}", headers=bob)
                for memory_id in sorted(alice_ids)
            ]
            ok = (
                all(r.status_code == 404 for r in responses)
                and not (alice_ids & bob_ids)
                and all(r.status_code == 404 for r in foreign)
            )
            checks += [
                (index, c) for c in check_step(step.expect, Observed(200, probe_ok=ok), mode)
            ]
            continue
        assert step.user is not None
        key = f"{scenario.id}:{index}"
        started = time.monotonic()
        response = await client.post(
            f"/v1/conversations/{conversation}/messages",
            json={"content": step.user, "idempotency_key": key},
            headers=alice,
        )
        seen = Observed(status=response.status_code, message=step.user)
        invalid = 0
        if response.status_code in (200, 201):
            body = response.json()
            seen.reply = body["assistant_content"]
            seen.safety_pathway = body["safety"]["pathway"] if body.get("safety") else None
            seen.sources = body.get("sources") or []
            async with app.state.session_factory() as session:
                metadata = await session.scalar(
                    text("SELECT generation_metadata FROM messages WHERE id = :id"),
                    {"id": uuid.UUID(body["assistant_message_id"])},
                )
            knowledge = (metadata or {}).get("knowledge") or {}
            seen.knowledge_status = knowledge.get("status")
            seen.knowledge_retrieve = knowledge.get("retrieve")
            seen.passages = [p["source_key"] for p in knowledge.get("passages") or []]
            invalid = knowledge.get("invalid_citations", 0)
        results = check_step(step.expect, seen, mode)
        checks += [(index, c) for c in results]
        transcript.append(
            {
                "step": index,
                "user": step.user,
                "status": seen.status,
                "reply": seen.reply,
                "safety_pathway": seen.safety_pathway,
                "knowledge": seen.knowledge_status,
                "passages": seen.passages,
                "cited": [s.get("source_key") for s in seen.sources],
                # Markers the model used that were not given (removed before storing).
                "invalid_citations": invalid,
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        if mode == "live":
            await asyncio.sleep(pause)
    return ScenarioOutcome(scenario, checks), transcript


async def run(args: argparse.Namespace) -> int:
    base = Settings()
    scenarios = load_scenarios(DATASET.read_text(encoding="utf-8").splitlines())
    scenarios = [
        s
        for s in scenarios
        if args.mode in s.modes
        and (args.split == "all" or s.split == args.split)
        and (not args.only or s.id in set(args.only))
    ]
    if args.limit:
        scenarios = scenarios[: args.limit]
    mode: Mode = args.mode
    outcomes: list[ScenarioOutcome] = []
    replies: dict[str, Any] = {}
    started = time.monotonic()
    async with temp_database(base.database_url) as url:
        settings = settings_for(mode, base, url)
        embeddings = build_embedding_client(settings)
        if embeddings is not None:
            await ready(embeddings)
        engine = create_async_engine(url)
        report = await ingest(
            load_manifest(),
            KnowledgeStore(async_sessionmaker(engine, expire_on_commit=False)),
            params=chunk_params(settings),
            embeddings=embeddings,
        )
        await engine.dispose()
        if embeddings is not None:
            await embeddings.aclose()
        if report.failed:
            raise SystemExit(f"corpus ingestion failed: {[s.key for s in report.failed]}")
        resolved = resolve_provider(settings)
        inner: LLMClient = build_llm_client(settings, resolved)
        if mode == "live":
            inner = CachingLLM(inner, CACHE_DIR / f"golden-{resolved.model.replace('/', '_')}.json")
        llm = FaultInjector(inner)
        safety_llm: FaultInjector | None = None
        provider = classifier_provider(settings)
        if provider is not None:
            resolved_safety = resolve_safety_llm(settings, provider)
            safety_inner: LLMClient = build_llm_client(settings, resolved_safety)
            if mode == "live":
                safety_inner = CachingLLM(
                    safety_inner,
                    CACHE_DIR / f"golden-{resolved_safety.model.replace('/', '_')}.json",
                )
            safety_llm = FaultInjector(safety_inner)
        knowledge_fault = KnowledgeFault()
        knowledge_fault.install()
        app = create_app(settings, llm_client=llm, safety_llm=safety_llm)
        try:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://golden", timeout=120
                ) as client:
                    for n, scenario in enumerate(scenarios, 1):
                        try:
                            outcome, transcript = await run_scenario(
                                scenario,
                                mode,
                                client,
                                app,
                                llm,
                                safety_llm,
                                knowledge_fault,
                                args.pause,
                            )
                        except Exception as exc:  # recorded, the run goes on
                            outcome = ScenarioOutcome(scenario, [], error=type(exc).__name__)
                            transcript = []
                        outcomes.append(outcome)
                        replies[scenario.id] = transcript
                        state = {True: "ok", False: "FAIL", None: "unchecked"}[outcome.passed]
                        failed = [
                            f"{i}:{c.name}({c.detail})"
                            for i, c in outcome.checks
                            if c.outcome == "fail"
                        ]
                        print(f"[{n}/{len(scenarios)}] {scenario.id} {state} {' '.join(failed)}")
        finally:
            knowledge_fault.uninstall()
            if safety_llm is not None:
                await safety_llm.aclose()
    summary = summarize(outcomes)
    metadata = experiment_metadata(
        settings, suite=f"golden-{mode}", dataset=DATASET, limitations=LIMITATIONS
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"golden-{mode}-{args.split}-{stamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval": f"golden-{mode}",
        "split": args.split,
        "metadata": metadata,
        "summary": summary,
        "seconds": round(time.monotonic() - started, 1),
        "scenarios": [
            {
                "id": o.scenario.id,
                "category": o.scenario.category,
                "passed": o.passed,
                "error": o.error,
                "checks": [{"step": i, **asdict(c)} for i, c in o.checks],
            }
            for o in outcomes
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replies_path = out.with_suffix(".replies.json")
    replies_path.write_text(
        json.dumps(replies, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.record_db:
        engine = create_async_engine(base.database_url)
        recorder = RunRecorder(async_sessionmaker(engine, expire_on_commit=False))
        run_id = await recorder.start(f"golden-{mode}", out.name, metadata)
        await recorder.add(
            run_id,
            [
                ItemResult(
                    o.scenario.id,
                    o.passed,
                    {"checks": len(o.checks)},
                    {"category": o.scenario.category},
                    o.error,
                )
                for o in outcomes
            ],
        )
        await recorder.finish(run_id, summary)
        await engine.dispose()
        print(f"recorded run {run_id}")
    print(json.dumps({k: summary[k] for k in ("scenarios", "checked", "pass_rate", "errors")}))
    print(f"written: {out.relative_to(ROOT)}")
    return 0 if summary["errors"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["deterministic", "live"], default="deterministic")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--only", nargs="*", help="scenario ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pause", type=float, default=0.0, help="live: seconds between turns")
    parser.add_argument("--record-db", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
