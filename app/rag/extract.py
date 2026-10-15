"""HTML to structured text (spec §11.4 parsing and normalization).

Standard library only (html.parser). The page is parsed into a small element
tree, the main content is chosen with simple selectors, and the text is read
out as a sequence of headings and paragraphs, so chunking can follow the
page's structure (§11.5).

Selectors are a small CSS subset: `tag`, `#id`, `.class`, `[attr]`,
`[attr=value]`, combined on one element (`div#main.body`), joined by
descendant (space) or child (`>`) combinators, in comma-separated lists. No
sibling combinators or pseudo-classes. Each extraction profile names its
content selectors (every matching element, in document order) and the parts
to drop (navigation, download links, feedback forms).

Normalization: NFKC, no control or zero-width characters, whitespace
collapsed. The checksum is taken over the normalized structure, so cosmetic
markup changes on the source page do not create a new document version.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Never content.
ALWAYS_DROP = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "form",
        "button",
        "select",
        "nav",
        "head",
    }
)
_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Elements whose text forms one paragraph.
_BLOCKS = frozenset(
    {
        "p",
        "li",
        "dt",
        "dd",
        "blockquote",
        "pre",
        "td",
        "th",
        "figcaption",
        "caption",
        "summary",
        "address",
    }
)
# Containers: text directly inside them is flushed as its own paragraph.
_CONTAINERS = frozenset(
    {
        "div",
        "section",
        "article",
        "main",
        "aside",
        "header",
        "footer",
        "ul",
        "ol",
        "dl",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "details",
        "figure",
        "body",
        "html",
    }
)
# A start tag that implicitly closes an open element of these kinds.
_CLOSES: dict[str, frozenset[str]] = {
    "p": frozenset(_BLOCKS | _CONTAINERS | set(HEADINGS)),
    "li": frozenset({"li"}),
    "dt": frozenset({"dt", "dd"}),
    "dd": frozenset({"dt", "dd"}),
    "tr": frozenset({"tr"}),
    "td": frozenset({"td", "th", "tr"}),
    "th": frozenset({"td", "th", "tr"}),
    "option": frozenset({"option"}),
}

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


class ExtractionError(Exception):
    """The page has no usable content (no selector matched, or no text)."""


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = "".join(c if unicodedata.category(c)[0] != "C" else " " for c in text)
    return " ".join(text.split())


# --- selectors ----------------------------------------------------------------

_COMPOUND = re.compile(
    r"^(?P<tag>[a-z][a-z0-9]*|\*)?(?P<rest>(?:[#.][\w-]+|\[[\w-]+(?:=[^\]]*)?\])*)$"
)
_PART = re.compile(r"#(?P<id>[\w-]+)|\.(?P<cls>[\w-]+)|\[(?P<attr>[\w-]+)(?:=(?P<value>[^\]]*))?\]")


@dataclass(frozen=True)
class Compound:
    """One element's conditions: `div#main.body[role=main]`."""

    tag: str | None
    ids: tuple[str, ...]
    classes: tuple[str, ...]
    attrs: tuple[tuple[str, str | None], ...]

    @classmethod
    def parse(cls, text: str) -> Compound:
        match = _COMPOUND.match(text)
        if match is None or not text:
            raise ValueError(f"unsupported selector: {text!r}")
        ids: list[str] = []
        classes: list[str] = []
        attrs: list[tuple[str, str | None]] = []
        for part in _PART.finditer(match.group("rest")):
            if part.group("id"):
                ids.append(part.group("id"))
            elif part.group("cls"):
                classes.append(part.group("cls"))
            else:
                value = part.group("value")
                unquoted = value.strip("\"'") if value is not None else None
                attrs.append((part.group("attr"), unquoted))
        tag = match.group("tag")
        return cls(None if tag in (None, "*") else tag, tuple(ids), tuple(classes), tuple(attrs))

    def matches(self, element: Element) -> bool:
        if self.tag is not None and element.tag != self.tag:
            return False
        if any(element.attrs.get("id") != i for i in self.ids):
            return False
        classes = set((element.attrs.get("class") or "").split())
        if not set(self.classes) <= classes:
            return False
        for name, value in self.attrs:
            if name not in element.attrs:
                return False
            if value is not None and element.attrs[name] != value:
                return False
        return True


@dataclass(frozen=True)
class Selector:
    """Compounds joined by descendant (space) or child (`>`) combinators,
    matched right to left: `main#maincontent > article`, `#topic .title h1`."""

    # (compound, combinator linking it to the previous step: " ", ">" or "").
    steps: tuple[tuple[Compound, str], ...]

    @classmethod
    def parse(cls, text: str) -> Selector:
        steps: list[tuple[Compound, str]] = []
        combinator = ""
        for part in text.replace(">", " > ").split():
            if part == ">":
                if not steps or combinator == ">":
                    raise ValueError(f"unsupported selector: {text!r}")
                combinator = ">"
                continue
            steps.append((Compound.parse(part), combinator if steps else ""))
            combinator = " "
        if not steps or combinator == ">":
            raise ValueError(f"unsupported selector: {text!r}")
        return cls(tuple(steps))

    def matches(self, element: Element) -> bool:
        return self._match(len(self.steps) - 1, element)

    def _match(self, index: int, element: Element) -> bool:
        compound, combinator = self.steps[index]
        if not compound.matches(element):
            return False
        if index == 0:
            return True
        if combinator == ">":
            return element.parent is not None and self._match(index - 1, element.parent)
        ancestor = element.parent
        while ancestor is not None:
            if self._match(index - 1, ancestor):
                return True
            ancestor = ancestor.parent
        return False


def parse_selectors(texts: Iterable[str]) -> tuple[Selector, ...]:
    """Each text may be a comma-separated list, as in CSS."""
    return tuple(
        Selector.parse(part.strip()) for t in texts for part in t.split(",") if part.strip()
    )


# --- tree -------------------------------------------------------------------------


@dataclass(eq=False)
class Element:
    tag: str
    attrs: dict[str, str | None]
    children: list[Element | str] = field(default_factory=list)
    parent: Element | None = field(default=None, repr=False)

    def iter(self) -> Iterator[Element]:
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.iter()

    def text(self) -> str:
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            elif child.tag not in ALWAYS_DROP:
                parts.append(child.text())
                if child.tag == "br":
                    parts.append(" ")
        return "".join(parts)


class _TreeBuilder(HTMLParser):
    """Lenient: unknown end tags are ignored, unclosed elements are closed by
    the usual implicit rules (a new <p> closes an open <p>, and so on)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element("#root", {})
        self._stack: list[Element] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        closers = [t for t, closed_by in _CLOSES.items() if tag in closed_by]
        if closers:
            # Close the nearest open element this tag implicitly ends, but not
            # across a container boundary (a <li> inside a nested list).
            for index in range(len(self._stack) - 1, 0, -1):
                current = self._stack[index].tag
                if current in closers:
                    del self._stack[index:]
                    break
                if current in _CONTAINERS or current in HEADINGS or current in ALWAYS_DROP:
                    break
        element = Element(tag, dict(attrs), parent=self._stack[-1])
        self._stack[-1].children.append(element)
        if tag not in _VOID:
            self._stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack[-1].children.append(Element(tag, dict(attrs), parent=self._stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def parse_html(html: str) -> Element:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


# --- structured text ----------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A heading (level 1-6) or a paragraph (level 0)."""

    level: int
    text: str


@dataclass(frozen=True)
class ExtractedPage:
    title: str
    blocks: tuple[Block, ...]

    @property
    def checksum(self) -> str:
        body = "\n".join(f"{'#' * b.level} {b.text}" if b.level else b.text for b in self.blocks)
        return hashlib.sha256(body.encode()).hexdigest()

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks if b.level == 0)


@dataclass(frozen=True)
class ExtractRules:
    content: tuple[Selector, ...]
    drop: tuple[Selector, ...] = ()
    # Headings (case-insensitive) whose sections are left out, with everything
    # under them: "References", "See also", "Related pages".
    stop_headings: frozenset[str] = frozenset()
    # Paragraphs (or list items) whose whole text matches one of these are
    # left out: "This page is also available in Spanish", "Download PDF".
    drop_text: tuple[re.Pattern[str], ...] = ()
    # Paragraphs shorter than this many words are kept only inside lists.
    min_paragraph_words: int = 3

    @classmethod
    def from_lists(
        cls,
        content: Sequence[str],
        drop: Sequence[str] = (),
        stop_headings: Sequence[str] = (),
        drop_text: Sequence[str] = (),
        min_paragraph_words: int = 3,
    ) -> ExtractRules:
        return cls(
            content=parse_selectors(content),
            drop=parse_selectors(drop),
            stop_headings=frozenset(_heading_key(h) for h in stop_headings),
            drop_text=tuple(re.compile(t, re.IGNORECASE) for t in drop_text),
            min_paragraph_words=min_paragraph_words,
        )


def _heading_key(text: str) -> str:
    return normalize_text(text).lower().rstrip(":").strip()


def _hidden(element: Element) -> bool:
    return "hidden" in element.attrs or element.attrs.get("aria-hidden") == "true"


class _Reader:
    def __init__(self, rules: ExtractRules) -> None:
        self._rules = rules
        self.blocks: list[Block] = []
        self._buffer: list[str] = []
        self._list_depth = 0
        # Level of a heading whose section is being skipped (stop_headings).
        self._skipping: int | None = None

    def _dropped(self, element: Element) -> bool:
        return (
            element.tag in ALWAYS_DROP
            or _hidden(element)
            or any(s.matches(element) for s in self._rules.drop)
        )

    def _emit(self, block: Block) -> None:
        if block.level:
            if self._skipping is not None and block.level > self._skipping:
                return
            self._skipping = (
                block.level if _heading_key(block.text) in self._rules.stop_headings else None
            )
            if self._skipping is not None:
                return
        elif self._skipping is not None:
            return
        self.blocks.append(block)

    def _flush(self, *, in_list: bool = False) -> None:
        text = normalize_text("".join(self._buffer))
        self._buffer.clear()
        if not text:
            return
        if not in_list and len(text.split()) < self._rules.min_paragraph_words:
            return
        if any(pattern.fullmatch(text) for pattern in self._rules.drop_text):
            return
        self._emit(Block(0, f"- {text}" if in_list else text))

    def read(self, element: Element) -> None:
        for child in element.children:
            if isinstance(child, str):
                self._buffer.append(child)
                continue
            if self._dropped(child):
                continue
            tag = child.tag
            if tag in HEADINGS:
                self._flush()
                text = normalize_text(child.text())
                if text:
                    self._emit(Block(HEADINGS[tag], text))
            elif tag in _BLOCKS:
                self._flush()
                nested = any(
                    isinstance(c, Element) and (c.tag in _CONTAINERS or c.tag in _BLOCKS)
                    for c in child.children
                )
                in_list = tag == "li"
                if nested:
                    # A list item with a nested list: its own text first, then the
                    # nested items as their own paragraphs.
                    self._list_depth += in_list
                    self.read(child)
                    self._flush(in_list=in_list)
                    self._list_depth -= in_list
                else:
                    self._buffer.append(child.text())
                    self._flush(in_list=in_list)
            elif tag in _CONTAINERS:
                self._flush(in_list=self._list_depth > 0)
                outer = self._skipping
                self.read(child)
                self._flush(in_list=self._list_depth > 0)
                if tag == "aside":
                    # A sidebar's stop heading ("Share this") ends only the sidebar.
                    self._skipping = outer
            elif tag == "br":
                self._buffer.append(" ")
            else:
                self._buffer.append(child.text())


def _inside(element: Element, chosen: set[int]) -> bool:
    ancestor = element.parent
    while ancestor is not None:
        if id(ancestor) in chosen:
            return True
        ancestor = ancestor.parent
    return False


def extract(html: str, rules: ExtractRules) -> ExtractedPage:
    """The content is every element matching any content selector, in
    document order (an element inside another match is read once, as part of
    it). Raises ExtractionError when nothing matches or there is no text."""
    root = parse_html(html)
    matched: list[Element] = []
    chosen: set[int] = set()
    for element in root.iter():
        if any(s.matches(element) for s in rules.content) and not _inside(element, chosen):
            matched.append(element)
            chosen.add(id(element))
    if not matched:
        raise ExtractionError("no content selector matched")
    title_element = next((e for e in root.iter() if e.tag == "title"), None)
    title = normalize_text(title_element.text()) if title_element is not None else ""
    reader = _Reader(rules)
    # A synthetic parent, so a matched heading or paragraph is read as one.
    reader.read(Element("#content", {}, children=list(matched)))
    reader._flush()
    blocks = _without_trailing_headings(reader.blocks)
    if not any(b.level == 0 for b in blocks):
        raise ExtractionError("content has no text")
    return ExtractedPage(title=title, blocks=tuple(blocks))


def _without_trailing_headings(blocks: list[Block]) -> list[Block]:
    """Headings with no paragraph before the next heading of the same or a
    higher level are dropped ("Contents", empty accordions)."""
    kept: list[Block] = []
    for index, block in enumerate(blocks):
        if block.level:
            following = next(
                (b for b in blocks[index + 1 :] if b.level == 0 or b.level <= block.level),
                None,
            )
            if following is None or following.level != 0:
                continue
        kept.append(block)
    return kept
