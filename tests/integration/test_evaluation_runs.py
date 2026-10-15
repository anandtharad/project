"""Evaluation runs are recorded with their metadata and per-item results
(Phase 6: failures trace to component, model, prompt and retrieval versions)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.evaluation.metadata import diff, experiment_metadata
from app.evaluation.runs import ItemResult, RunRecorder
from tests.helpers import make_settings


@pytest.fixture
async def recorder(db_engine: AsyncEngine) -> AsyncIterator[RunRecorder]:
    async with db_engine.begin() as conn:
        await conn.execute(text("TRUNCATE evaluation_runs CASCADE"))
    yield RunRecorder(async_sessionmaker(db_engine, expire_on_commit=False))


async def test_a_run_is_recorded_with_metadata_and_results(recorder: RunRecorder) -> None:
    metadata = experiment_metadata(make_settings(), suite="rag-retrieval")
    run_id = await recorder.start("rag-retrieval", "dev 180/30/25", metadata)
    await recorder.add(
        run_id,
        [
            ItemResult("rq-001", True, {"rank": 1}, {"type": "answerable"}),
            ItemResult("rq-002", False, {"rank": None}, {"type": "answerable"}, "x" * 500),
        ],
    )
    # A resumed run records an item again: it is replaced, not duplicated.
    await recorder.add(run_id, [ItemResult("rq-002", True, {"rank": 3})])
    await recorder.finish(run_id, {"hit_rate": 1.0})

    run = await recorder.get(run_id)
    assert run is not None and run.status == "completed"
    assert run.metrics == {"hit_rate": 1.0}
    assert run.metadata["prompts"]["companion"]["version"]
    assert run.metadata["knowledge"]["manifest_sha256"]
    assert [r.item_id for r in run.results] == ["rq-001", "rq-002"]
    assert run.results[1].passed is True and run.results[1].metrics == {"rank": 3}
    assert await recorder.latest("rag-retrieval") == [(run_id, "dev 180/30/25", "completed")]
    assert await recorder.delete_run(run_id)
    assert await recorder.get(run_id) is None


def test_metadata_records_versions_and_never_secrets() -> None:
    settings = make_settings(groq_api_key="gsk-secret-value-123")
    metadata = experiment_metadata(settings, suite="x", limitations=["synthetic data"])
    assert "gsk-secret-value-123" not in repr(metadata)
    assert metadata["limitations"] == ["synthetic data"]
    assert metadata["decisions"]["questions"]
    other = experiment_metadata(make_settings(knowledge_min_semantic=0.65), suite="x")
    changes = {path for path, _, _ in diff(metadata, other)}
    assert "knowledge.min_semantic" in changes and "created_at" not in changes
