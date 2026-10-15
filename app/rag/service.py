"""The knowledge lookup for one turn: route, retrieve, evidence (spec §11.7-11.9).

It cannot fail a turn: a store or embedding failure, or the lookup's own
time limit, ends in status `unavailable`, and the reply goes ahead without
evidence and without claiming a source was consulted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from app.llm.schemas import ChatMessage, ChatRole
from app.observability import log_event
from app.rag.evidence import EvidenceItem, EvidenceStatus, number, render_evidence
from app.rag.retrieval import KnowledgeRetriever, RetrievalOutcome
from app.rag.routing import Route, route
from app.rag.store import NO_FILTER, KnowledgeFilter

logger = logging.getLogger("app.rag")

LookupStatus = Literal[
    "not_needed", "ok", "no_results", "unavailable", "skipped_safety", "disabled"
]
# Routes where "no source" is said rather than left unsaid.
_ASKED_FOR_SOURCES = frozenset({"evidence", "source_followup", "current_info"})


@dataclass(frozen=True)
class KnowledgeLookup:
    route: Route
    status: LookupStatus
    items: tuple[EvidenceItem, ...] = ()
    outcome: RetrievalOutcome | None = None

    def skipped_for_safety(self) -> KnowledgeLookup:
        """Safety pathways never carry evidence (docs/rag.md)."""
        if not self.route.retrieve:
            return self
        return replace(self, status="skipped_safety", items=())

    def evidence_block(self) -> str | None:
        if self.items:
            return render_evidence(
                self.items, "ok", current_info_requested=self.route.kind == "current_info"
            )
        if self.route.kind in _ASKED_FOR_SOURCES and self.status in ("no_results", "unavailable"):
            status: EvidenceStatus = (
                "unavailable" if self.status == "unavailable" else "no_relevant_sources"
            )
            return render_evidence((), status)
        return None

    def summary(self) -> dict[str, Any]:
        outcome = self.outcome
        return {
            "route": self.route.kind,
            "reason": self.route.reason,
            "retrieve": self.route.retrieve,
            "status": self.status,
            "mode": outcome.mode if outcome is not None else None,
            "failure": outcome.failure if outcome is not None else None,
            "considered": outcome.considered if outcome is not None else 0,
            "latency_ms": outcome.latency_ms if outcome is not None else None,
            "passages": [
                {
                    "marker": item.marker,
                    "chunk_id": str(item.passage.candidate.chunk_id),
                    "document_id": str(item.passage.candidate.document_id),
                    "source_key": item.passage.candidate.source_key,
                    "version": item.passage.candidate.version,
                    "semantic": (
                        round(item.passage.candidate.semantic, 4)
                        if item.passage.candidate.semantic is not None
                        else None
                    ),
                    "semantic_rank": item.passage.semantic_rank,
                    "lexical_rank": item.passage.lexical_rank,
                }
                for item in self.items
            ],
        }


def previous_user_message(history: Sequence[ChatMessage]) -> str | None:
    """The person's message before the current one (history ends with it)."""
    for message in reversed(history[:-1]):
        if message.role is ChatRole.USER:
            return message.content
    return None


class KnowledgeService:
    def __init__(
        self,
        retriever: KnowledgeRetriever,
        *,
        timeout_seconds: float,
        filters: KnowledgeFilter = NO_FILTER,
    ) -> None:
        self.retriever = retriever
        self._timeout = timeout_seconds
        self._filters = filters

    async def lookup(self, message: str, history: Sequence[ChatMessage]) -> KnowledgeLookup:
        chosen = route(message, previous_user_message=previous_user_message(history))
        if not chosen.retrieve:
            return KnowledgeLookup(chosen, "not_needed")
        try:
            async with asyncio.timeout(self._timeout):
                outcome = await self.retriever.retrieve(chosen.query, self._filters)
        except TimeoutError:
            log_event(logger, "knowledge.timeout", level=logging.WARNING)
            return KnowledgeLookup(chosen, "unavailable")
        except Exception as exc:  # retrieval must never fail the turn
            log_event(
                logger,
                "knowledge.failed",
                level=logging.WARNING,
                exc_type=type(exc).__name__,
            )
            return KnowledgeLookup(chosen, "unavailable")
        if outcome.status == "unavailable":
            return KnowledgeLookup(chosen, "unavailable", outcome=outcome)
        items = tuple(number(outcome.passages))
        return KnowledgeLookup(chosen, "ok" if items else "no_results", items, outcome)
