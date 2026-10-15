"""Acquire knowledge sources: download each page in the manifest, extract its
main content and write the snapshot that ingestion reads (docs/rag.md).

Only the extracted text and headings are stored (data/knowledge/sources/),
never the page itself. robots.txt is checked before every download, the
User-Agent names no person, and requests to one host are spaced out.

Usage:
  uv run python scripts/fetch_knowledge.py                 # every source
  uv run python scripts/fetch_knowledge.py --only cdc-sleep-about
  uv run python scripts/fetch_knowledge.py --raw-dir DIR   # saved pages: DIR/<key>.html
  uv run python scripts/fetch_knowledge.py --check         # re-extract, report changes only
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.extract import ExtractionError, extract
from app.rag.manifest import (
    DEFAULT_MANIFEST,
    Manifest,
    ManifestError,
    Snapshot,
    SnapshotError,
    SourceEntry,
    load_manifest,
    read_snapshot,
    write_snapshot,
)

USER_AGENT = "companion-ai-knowledge-fetch/1.0 (non-commercial; reviewed-corpus snapshot)"
MAX_BYTES = 5_000_000


class FetchError(Exception):
    pass


def _robots(url: str, cache: dict[str, urllib.robotparser.RobotFileParser]) -> bool:
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    parser = cache.get(base)
    if parser is None:
        parser = urllib.robotparser.RobotFileParser(f"{base}/robots.txt")
        robots_url = f"{base}/robots.txt"  # https: entries are validated by the manifest
        request = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - https only
                parser.parse(response.read(MAX_BYTES).decode("utf-8", "replace").splitlines())
        except urllib.error.HTTPError as exc:
            # As robotparser.read() does: 401/403 disallow everything, other errors allow.
            parser.parse(["User-agent: *", "Disallow: /"] if exc.code in (401, 403) else [])
        cache[base] = parser
    return parser.can_fetch(USER_AGENT, url)


def download(url: str) -> str:
    if not url.startswith("https://"):
        raise FetchError("only https sources are fetched")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - https only
            body = response.read(MAX_BYTES + 1)
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(f"network error: {type(exc).__name__}") from exc
    if len(body) > MAX_BYTES:
        raise FetchError("page too large")
    text: str = body.decode(charset, "replace")
    return text


def acquire(
    entry: SourceEntry,
    manifest: Manifest,
    *,
    raw_dir: Path | None,
    robots: dict[str, urllib.robotparser.RobotFileParser],
) -> Snapshot:
    key = entry.meta.key
    if raw_dir is not None:
        path = raw_dir / f"{key}.html"
        html = path.read_text(encoding="utf-8", errors="replace")
        fetched = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    else:
        if not _robots(entry.meta.url, robots):
            raise FetchError("disallowed by robots.txt")
        html = download(entry.meta.url)
        fetched = datetime.now(UTC)
    page = extract(html, manifest.profiles[entry.profile].rules())
    return Snapshot(
        key=key,
        url=entry.meta.url,
        fetched_at=fetched.isoformat(timespec="seconds"),
        raw_sha256=hashlib.sha256(html.encode()).hexdigest(),
        page=page,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", nargs="*", help="source keys")
    parser.add_argument("--raw-dir", type=Path, help="use saved pages DIR/<key>.html")
    parser.add_argument("--check", action="store_true", help="report changes, write nothing")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        for problem in exc.problems:
            print(f"manifest: {problem}")
        return 2
    wanted = set(args.only or [])
    unknown = wanted - {s.meta.key for s in manifest.sources}
    if unknown:
        print(f"unknown keys: {sorted(unknown)}")
        return 2
    robots: dict[str, urllib.robotparser.RobotFileParser] = {}
    last_request: dict[str, float] = {}
    failures = 0
    for entry in manifest.sources:
        key = entry.meta.key
        if wanted and key not in wanted:
            continue
        host = urlsplit(entry.meta.url).netloc
        if args.raw_dir is None and host in last_request:
            time.sleep(max(0.0, args.delay - (time.monotonic() - last_request[host])))
        try:
            snapshot = acquire(entry, manifest, raw_dir=args.raw_dir, robots=robots)
        except (FetchError, ExtractionError, OSError) as exc:
            failures += 1
            print(f"FAIL  {key}: {exc}")
            continue
        finally:
            last_request[host] = time.monotonic()
        try:
            previous = read_snapshot(entry.snapshot)
        except SnapshotError:
            previous = None
        changed = previous is None or previous.page.checksum != snapshot.page.checksum
        status = "changed" if changed else "same"
        if not args.check and changed:
            write_snapshot(entry.snapshot, snapshot)
        blocks, words = len(snapshot.page.blocks), snapshot.page.word_count
        print(f"{status:<8}{key}: {blocks} blocks, {words} words")
    print(f"done: {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
