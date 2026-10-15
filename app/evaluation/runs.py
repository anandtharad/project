"""Recording evaluation runs (spec §15.1, §19.3; Phase 6).

A run is stored in `evaluation_runs` with its experiment metadata
(app/evaluation/metadata.py) and summary metrics, and each item's outcome in
`evaluation_results`: ids, labels, metrics and a pass flag, never
conversation text. Result files in eval/results/ remain the primary record
for the evaluation scripts; the tables make runs queryable and comparable
(scripts/eval_report.py).

Runs and results are evaluation artifacts: they hold no personal data
(evaluation sets are synthetic or public) and are deleted with
`delete_run`; see docs/evaluation.md for retention.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EvaluationResult, EvaluationRun

_MAX_ERROR = 200


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    passed: bool | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class StoredRun:
    id: uuid.UUID
    suite: str
    name: str
    status: str
    metadata: dict[str, Any]
    metrics: dict[str, Any] | None
    results: tuple[ItemResult, ...]


class RunRecorder:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def start(self, suite: str, name: str, metadata: Mapping[str, Any]) -> uuid.UUID:
        async with self._sessions() as session, session.begin():
            run = EvaluationRun(suite=suite, name=name[:200], run_metadata=dict(metadata))
            session.add(run)
            await session.flush()
            return run.id

    async def add(self, run_id: uuid.UUID, results: Iterable[ItemResult]) -> int:
        """Insert or replace items (a resumed run may record an item again)."""
        rows = [
            {
                "run_id": run_id,
                "item_id": r.item_id[:128],
                "passed": r.passed,
                "metrics": dict(r.metrics),
                "labels": dict(r.labels),
                "error": r.error[:_MAX_ERROR] if r.error else None,
            }
            for r in results
        ]
        if not rows:
            return 0
        statement = pg_insert(EvaluationResult).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="uq_evaluation_results_run_item",
            set_={
                "passed": statement.excluded.passed,
                "metrics": statement.excluded.metrics,
                "labels": statement.excluded.labels,
                "error": statement.excluded.error,
            },
        )
        async with self._sessions() as session, session.begin():
            await session.execute(statement)
        return len(rows)

    async def finish(
        self, run_id: uuid.UUID, metrics: Mapping[str, Any], *, failed: bool = False
    ) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(EvaluationRun)
                .where(EvaluationRun.id == run_id)
                .values(
                    status="failed" if failed else "completed",
                    metrics=dict(metrics),
                    finished_at=func.now(),
                )
            )

    async def get(self, run_id: uuid.UUID) -> StoredRun | None:
        async with self._sessions() as session:
            run = await session.get(EvaluationRun, run_id)
            if run is None:
                return None
            rows = await session.scalars(
                select(EvaluationResult)
                .where(EvaluationResult.run_id == run_id)
                .order_by(EvaluationResult.item_id)
            )
            return StoredRun(
                id=run.id,
                suite=run.suite,
                name=run.name,
                status=run.status,
                metadata=run.run_metadata,
                metrics=run.metrics,
                results=tuple(
                    ItemResult(r.item_id, r.passed, r.metrics, r.labels, r.error) for r in rows
                ),
            )

    async def latest(self, suite: str, *, limit: int = 10) -> list[tuple[uuid.UUID, str, str]]:
        """(id, name, status) of the suite's most recent runs."""
        async with self._sessions() as session:
            rows = await session.execute(
                select(EvaluationRun.id, EvaluationRun.name, EvaluationRun.status)
                .where(EvaluationRun.suite == suite)
                .order_by(EvaluationRun.started_at.desc())
                .limit(limit)
            )
            return [(r[0], r[1], r[2]) for r in rows]

    async def delete_run(self, run_id: uuid.UUID) -> bool:
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                delete(EvaluationRun).where(EvaluationRun.id == run_id).returning(EvaluationRun.id)
            )
            return result.first() is not None
