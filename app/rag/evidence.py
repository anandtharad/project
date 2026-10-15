"""Grounded generation (spec §11.8, §13): the evidence block and citations.

Retrieved passages are untrusted. They travel in the current user message,
inside an <evidence> block of JSON whose "<" and ">" are escaped, so nothing
in a passage can close the block or pose as an instruction. Each passage has
a marker (S1, S2, ...) that the reply may cite as [S1].

After generation, every marker in the reply is checked against the passages
that were actually given. Valid ones resolve to a chunk and its document;
invalid ones (a number that was not given, or any marker when no evidence
was given) are removed from the reply and counted, so a citation shown to
the person always resolves. A removed marker leaves its sentence as general
information, which the prompt already allows.

On an evidence request with no relevant passage, the block is sent with no
passages and a status, so the reply can say it has no source to hand
instead of pretending to have looked something up (§11.9).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.rag.retrieval import Passage

EvidenceStatus = Literal["ok", "no_relevant_sources", "unavailable"]

# [S1], [S1, S2], [S1][S2]; also tolerant of "[s1]" and "[S 1]".
_MARKERS = re.compile(r"\[\s*([Ss]\s*\d+(?:\s*[,;]\s*[Ss]\s*\d+)*)\s*\]")
_ONE = re.compile(r"[Ss]\s*(\d+)")
# Markers in earlier replies refer to earlier evidence blocks; they are
# removed from the history the model sees, so S1 always means this turn's S1.
_ANY_MARKER = re.compile(r"\s?\[\s*[Ss]\s*\d+(?:\s*[,;]\s*[Ss]\s*\d+)*\s*\]")


def _escaped_json(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, indent=1)
    return body.replace("<", "\\u003c").replace(">", "\\u003e")


@dataclass(frozen=True)
class EvidenceItem:
    marker: str
    passage: Passage


def number(passages: Sequence[Passage]) -> list[EvidenceItem]:
    return [EvidenceItem(f"S{i}", p) for i, p in enumerate(passages, start=1)]


def render_evidence(
    items: Sequence[EvidenceItem],
    status: EvidenceStatus = "ok",
    *,
    current_info_requested: bool = False,
) -> str:
    body: dict[str, Any] = {
        "status": status,
        "passages": [
            {
                "marker": item.marker,
                "source": item.passage.candidate.title,
                "publisher": item.passage.candidate.publisher,
                "section": item.passage.candidate.heading,
                "reviewed": (
                    item.passage.candidate.reviewed_on.isoformat()
                    if item.passage.candidate.reviewed_on
                    else None
                ),
                "text": item.passage.candidate.text,
            }
            for item in items
        ],
    }
    if current_info_requested:
        body["note"] = "The person asked for current information; the library is offline."
    return f"<evidence>\n{_escaped_json(body)}\n</evidence>"


@dataclass(frozen=True)
class Citation:
    marker: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_key: str
    version: int
    title: str
    publisher: str
    url: str
    section: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "source_key": self.source_key,
            "version": self.version,
            "title": self.title,
            "publisher": self.publisher,
            "url": self.url,
            "section": self.section,
        }


@dataclass(frozen=True)
class CitationCheck:
    content: str
    citations: tuple[Citation, ...]
    # Marker occurrences that did not resolve and were removed.
    invalid: int
    # Markers given in the evidence block but not cited.
    uncited: tuple[str, ...]


def check_citations(reply: str, items: Sequence[EvidenceItem]) -> CitationCheck:
    by_marker = {item.marker: item for item in items}
    cited: dict[str, Citation] = {}
    invalid = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal invalid
        valid: list[str] = []
        for n in _ONE.findall(match.group(1)):
            marker = f"S{int(n)}"
            item = by_marker.get(marker)
            if item is None:
                invalid += 1
                continue
            if marker not in valid:
                valid.append(marker)
            candidate = item.passage.candidate
            cited.setdefault(
                marker,
                Citation(
                    marker=marker,
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    source_key=candidate.source_key,
                    version=candidate.version,
                    title=candidate.title,
                    publisher=candidate.publisher,
                    url=candidate.url,
                    section=candidate.heading,
                ),
            )
        return "".join(f"[{m}]" for m in valid)

    content = _MARKERS.sub(replace, reply)
    if invalid:
        # Removing a marker can leave "word ." or a double space.
        content = re.sub(r"[ \t]+([.,;:!?])", r"\1", content)
        content = re.sub(r"[ \t]{2,}", " ", content).strip()
    order = sorted(cited, key=lambda m: int(m[1:]))
    return CitationCheck(
        content=content,
        citations=tuple(cited[m] for m in order),
        invalid=invalid,
        uncited=tuple(m for m in by_marker if m not in cited),
    )


def strip_markers(text: str) -> str:
    """An earlier reply without its citation markers (for the model's history)."""
    return _ANY_MARKER.sub("", text)
