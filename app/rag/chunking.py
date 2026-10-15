"""Structure-aware chunking (spec §11.5).

A page's blocks are grouped into sections by heading. Each section's
paragraphs are packed into chunks of at most `max_words`, never crossing a
section boundary, and consecutive chunks of one section share `overlap_words`
words. A paragraph longer than `max_words` is split at sentence ends (or, for
one very long sentence, at word boundaries). A section's last chunk
is folded into the one before when it would add fewer than `min_words` new
words, and a chunk still under `min_words` (a short section) joins its
previous or next neighbour in the same top-level section, keeping its
subheading in the text. So a chunk can exceed `max_words` by less than
`min_words`, and a chunk under `min_words` remains only where it has no
neighbour in its top-level section.

Every chunk keeps its heading path ("Title > Section > Subsection"), so it
reads on its own in a prompt and its full-text vector includes the headings.
The default size (180 words, about 240 tokens) stays well under the local
embedding model's 512-token limit, beyond which text is cut off silently.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.extract import Block

CHUNKER_VERSION = "1"  # bump when the algorithm changes: every source is re-chunked
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class ChunkParams:
    max_words: int = 180
    overlap_words: int = 30
    min_words: int = 25

    def __post_init__(self) -> None:
        if self.max_words < 20:
            raise ValueError("max_words must be at least 20")
        if not 0 <= self.overlap_words < self.max_words // 2:
            raise ValueError("overlap_words must be below half of max_words")
        if not 0 <= self.min_words <= self.max_words:
            raise ValueError("min_words must be between 0 and max_words")

    @property
    def identity(self) -> str:
        """Part of a document version's checksum: changing the settings
        re-chunks every document on the next ingestion."""
        return f"chunker{CHUNKER_VERSION}:{self.max_words}/{self.overlap_words}/{self.min_words}"


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    heading: str
    text: str
    # Heading path below the title; the first entry is the top-level section.
    path: tuple[str, ...] = ()

    @property
    def top(self) -> str:
        return self.path[0] if self.path else ""

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class _Section:
    path: tuple[str, ...]
    paragraphs: list[str]


def _sections(title: str, blocks: Sequence[Block]) -> list[_Section]:
    sections: list[_Section] = []
    path: list[tuple[int, str]] = []
    current = _Section((), [])
    for block in blocks:
        if block.level:
            if current.paragraphs:
                sections.append(current)
            # A heading closes every heading of its level or deeper.
            path = [(lvl, txt) for lvl, txt in path if lvl < block.level]
            if block.level == 1 and block.text == title:
                current = _Section(tuple(t for _, t in path), [])
                continue
            path.append((block.level, block.text))
            current = _Section(tuple(t for _, t in path), [])
        else:
            current.paragraphs.append(block.text)
    if current.paragraphs:
        sections.append(current)
    return sections


def _pieces(paragraph: str, max_words: int) -> list[str]:
    """A paragraph as pieces of at most max_words, split at sentence ends."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        while len(words) > max_words:
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if words:
            pieces.append(" ".join(words))
    return pieces


def _pack(paragraphs: Sequence[str], params: ChunkParams) -> list[str]:
    pieces = [p for paragraph in paragraphs for p in _pieces(paragraph, params.max_words)]
    # Each chunk: (overlap carried from the previous chunk, its own pieces).
    chunks: list[tuple[str, list[str]]] = []
    overlap, size = "", 0
    current: list[str] = []
    for piece in pieces:
        n = len(piece.split())
        if current and size + n > params.max_words:
            chunks.append((overlap, current))
            # Carry the last overlap_words words into the next chunk.
            words = " ".join([overlap, *current]).split()
            overlap = " ".join(words[-params.overlap_words :]) if params.overlap_words else ""
            current = []
            size = len(overlap.split())
            if size + n > params.max_words:
                overlap, size = "", 0
        current.append(piece)
        size += n
    if current:
        new_words = sum(len(p.split()) for p in current)
        if chunks and new_words < params.min_words:
            # A small tail is folded into the previous chunk rather than stored
            # as a chunk that is mostly overlap.
            previous_overlap, previous = chunks[-1]
            chunks[-1] = (previous_overlap, [*previous, *current])
        else:
            chunks.append((overlap, current))
    return ["\n".join([o, *c] if o else c) for o, c in chunks]


def _heading(title: str, path: tuple[str, ...]) -> str:
    return " > ".join((title, *path)) if title else " > ".join(path)


def _join(first: Chunk, second: Chunk, title: str) -> Chunk:
    """Two neighbouring chunks of one top-level section as one: the heading is
    their common path, and text below it keeps its own subheading."""
    common: list[str] = []
    for a, b in zip(first.path, second.path, strict=False):
        if a != b:
            break
        common.append(a)

    def labelled(chunk: Chunk) -> str:
        rest = chunk.path[len(common) :]
        return f"{rest[-1]}: {chunk.text}" if rest else chunk.text

    path = tuple(common)
    return Chunk(
        first.ordinal, _heading(title, path), f"{labelled(first)}\n{labelled(second)}", path
    )


def _merge_small(chunks: list[Chunk], params: ChunkParams, title: str) -> list[Chunk]:
    """A chunk under min_words joins its previous neighbour, or else its next
    one, within the same top-level section, if the result stays under
    max_words + min_words."""
    limit = params.max_words + params.min_words
    out: list[Chunk] = []
    pending: Chunk | None = None  # a small chunk waiting to join the next one
    for chunk in chunks:
        if pending is not None:
            if pending.top == chunk.top and pending.words + chunk.words <= limit:
                chunk = _join(pending, chunk, title)
            else:
                out.append(pending)
            pending = None
        if chunk.words < params.min_words:
            previous = out[-1] if out else None
            if (
                previous is not None
                and previous.top == chunk.top
                and previous.words + chunk.words <= limit
            ):
                out[-1] = _join(previous, chunk, title)
                continue
            pending = chunk
            continue
        out.append(chunk)
    if pending is not None:
        out.append(pending)
    return [Chunk(i, c.heading, c.text, c.path) for i, c in enumerate(out)]


def chunk_page(title: str, blocks: Sequence[Block], params: ChunkParams) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _sections(title, blocks):
        for text in _pack(section.paragraphs, params):
            chunks.append(Chunk(len(chunks), _heading(title, section.path), text, section.path))
    return _merge_small(chunks, params, title)


def embedding_text(heading: str, text: str) -> str:
    """What is embedded for a chunk: its heading path, then its text."""
    return f"{heading}\n{text}" if heading else text
