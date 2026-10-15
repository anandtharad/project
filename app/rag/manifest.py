"""The knowledge manifest and source snapshots (spec §11.2-11.4).

`data/knowledge/manifest.toml` lists every reviewed source: its provenance
and licence (§11.3), how to extract its text (an extraction profile), the
snapshot file, and optional probe queries that the ingestion's retrieval
tests run. Acquisition (scripts/fetch_knowledge.py) downloads a page,
extracts its main content and writes the snapshot; ingestion
(scripts/ingest_knowledge.py) works from snapshots only, so it is offline
and repeatable.

A snapshot holds only the extracted text and headings, not the page: no
navigation, images or third-party material from the page is stored, and the
raw page's SHA-256 is kept for provenance.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.rag.extract import Block, ExtractedPage, ExtractRules
from app.rag.store import SourceMetadata

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "manifest.toml"
SNAPSHOT_FORMAT = 1

# Topic tags (spec §11.2). A source's topics must come from this list, so
# source filtering by topic cannot silently miss a misspelt tag.
TOPICS = frozenset(
    {
        "stress",
        "sleep",
        "anxiety",
        "low_mood",
        "mindfulness",
        "relaxation",
        "emotional_regulation",
        "communication",
        "relationships",
        "loneliness",
        "grief",
        "behaviour_change",
        "physical_activity",
        "nutrition",
        "self_esteem",
        "wellbeing",
        "work",
        "study",
    }
)
_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ManifestError(Exception):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Profile:
    """How to extract a site's main content (app/rag/extract.py selectors).
    The manifest's [defaults] stop headings and drop texts apply to every
    profile, in addition to its own."""

    content: tuple[str, ...]
    drop: tuple[str, ...] = ()
    stop_headings: tuple[str, ...] = ()
    drop_text: tuple[str, ...] = ()

    def rules(self) -> ExtractRules:
        return ExtractRules.from_lists(
            self.content, self.drop, self.stop_headings, drop_text=self.drop_text
        )


@dataclass(frozen=True)
class SourceEntry:
    meta: SourceMetadata
    profile: str
    snapshot: Path
    # Queries for which this source should be retrieved (ingestion check).
    probes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Manifest:
    path: Path
    profiles: dict[str, Profile]
    sources: tuple[SourceEntry, ...]
    review_note: str = ""

    def source(self, key: str) -> SourceEntry:
        return next(s for s in self.sources if s.meta.key == key)


def _date(value: Any, name: str, problems: list[str], *, required: bool) -> date | None:
    if value is None:
        if required:
            problems.append(f"{name} is required")
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        problems.append(f"{name} is not a date: {value!r}")
        return None


def _text(raw: dict[str, Any], name: str, problems: list[str], where: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}: {name} is required")
        return ""
    return value.strip()


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise TypeError


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Manifest:
    """Parses and validates the whole manifest; every problem is reported at
    once (ManifestError.problems), not just the first."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if raw.get("manifest_version") != 1:
        problems.append("manifest_version must be 1")
    profiles: dict[str, Profile] = {}
    defaults = raw.get("defaults") or {}
    try:
        default_stops = _strings(defaults.get("stop_headings"))
        default_drop_text = _strings(defaults.get("drop_text"))
    except TypeError:
        problems.append("defaults: lists of strings expected")
        default_stops, default_drop_text = (), ()
    for name, body in (raw.get("profiles") or {}).items():
        try:
            profile = Profile(
                content=_strings(body.get("content")),
                drop=_strings(body.get("drop")),
                stop_headings=default_stops + _strings(body.get("stop_headings")),
                drop_text=default_drop_text + _strings(body.get("drop_text")),
            )
            profile.rules()  # selectors and patterns compile
        except (TypeError, ValueError, re.error) as exc:
            problems.append(f"profile {name}: {exc or 'lists of strings expected'}")
            continue
        if not profile.content:
            problems.append(f"profile {name}: content selectors are required")
        profiles[name] = profile

    sources: list[SourceEntry] = []
    seen_keys: set[str] = set()
    seen_urls: set[str] = set()
    for index, body in enumerate(raw.get("sources") or []):
        key = body.get("key", "")
        where = f"source {key or index}"
        if not isinstance(key, str) or not _KEY.match(key):
            problems.append(f"{where}: key must be lowercase words joined by hyphens")
        elif key in seen_keys:
            problems.append(f"{where}: duplicate key")
        seen_keys.add(str(key))
        url = _text(body, "url", problems, where)
        if url and not url.startswith("https://"):
            problems.append(f"{where}: url must be https")
        if url in seen_urls:
            problems.append(f"{where}: duplicate url")
        seen_urls.add(url)
        title = _text(body, "title", problems, where)
        publisher = _text(body, "publisher", problems, where)
        licence = _text(body, "licence", problems, where)
        licence_url = body.get("licence_url")
        if licence_url is not None and not str(licence_url).startswith("https://"):
            problems.append(f"{where}: licence_url must be https")
        attribution = body.get("attribution")
        published_on = _date(
            body.get("published_on"), f"{where}: published_on", problems, required=False
        )
        reviewed_on = _date(
            body.get("reviewed_on"), f"{where}: reviewed_on", problems, required=True
        )
        try:
            topics = _strings(body.get("topics"))
            probes = _strings(body.get("probes"))
        except TypeError:
            problems.append(f"{where}: topics and probes must be lists of strings")
            topics, probes = (), ()
        if not topics:
            problems.append(f"{where}: at least one topic is required")
        unknown = sorted(set(topics) - TOPICS)
        if unknown:
            problems.append(f"{where}: unknown topics {unknown}")
        profile = body.get("profile", "")
        if profile not in profiles:
            problems.append(f"{where}: unknown profile {profile!r}")
        snapshot = path.parent / str(body.get("snapshot") or f"sources/{key}.json")
        if reviewed_on is None:
            continue
        sources.append(
            SourceEntry(
                meta=SourceMetadata(
                    key=str(key),
                    url=url,
                    title=title,
                    publisher=publisher,
                    licence=licence,
                    licence_url=str(licence_url) if licence_url else None,
                    attribution=str(attribution) if attribution else None,
                    published_on=published_on,
                    reviewed_on=reviewed_on,
                    topics=tuple(sorted(set(topics))),
                ),
                profile=str(profile),
                snapshot=snapshot,
                probes=probes,
            )
        )
    if problems:
        raise ManifestError(problems)
    return Manifest(
        path=path,
        profiles=profiles,
        sources=tuple(sources),
        review_note=str(raw.get("review_note", "")),
    )


# --- snapshots -------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    key: str
    url: str
    fetched_at: str
    # SHA-256 of the downloaded page, for provenance only.
    raw_sha256: str
    page: ExtractedPage
    extra: dict[str, Any] = field(default_factory=dict)


class SnapshotError(Exception):
    pass


def write_snapshot(path: Path, snapshot: Snapshot) -> None:
    body = {
        "format": SNAPSHOT_FORMAT,
        "key": snapshot.key,
        "url": snapshot.url,
        "fetched_at": snapshot.fetched_at,
        "raw_sha256": snapshot.raw_sha256,
        "page_title": snapshot.page.title,
        "blocks": [[b.level, b.text] for b in snapshot.page.blocks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


def read_snapshot(path: Path) -> Snapshot:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SnapshotError("snapshot missing (run scripts/fetch_knowledge.py)") from exc
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"snapshot unreadable: {type(exc).__name__}") from exc
    if not isinstance(body, dict) or body.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError("unsupported snapshot format")
    try:
        blocks = tuple(Block(int(level), str(text)) for level, text in body["blocks"])
        return Snapshot(
            key=str(body["key"]),
            url=str(body["url"]),
            fetched_at=str(body["fetched_at"]),
            raw_sha256=str(body["raw_sha256"]),
            page=ExtractedPage(title=str(body.get("page_title", "")), blocks=blocks),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"snapshot malformed: {type(exc).__name__}") from exc
