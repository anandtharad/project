"""Helpers for blind model judgements (spec §16.5).

Items from several runs are given opaque ids and shuffled, so a judge cannot
tell which model or prompt produced a reply; the key that maps ids back is
written apart from the packets. Agreement between judges is reported with
every result (raw agreement and Cohen's kappa).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def blind(
    items: Sequence[Mapping[str, Any]], seed: int, prefix: str = "j"
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Shuffle `items` (each a dict with "packet" and "source") and give each
    an opaque id. Returns the packets (what the judge sees, with "jid") and
    the key (jid -> source)."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)  # noqa: S311 - ordering, not security
    width = len(str(len(items)))
    packets, key = [], {}
    for n, index in enumerate(order, start=1):
        jid = f"{prefix}{n:0{width}d}"
        packets.append({"jid": jid, **items[index]["packet"]})
        key[jid] = dict(items[index]["source"])
    return packets, key


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two raters over the same items (None when undefined:
    no items, or both raters always gave the same single label)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in set(count_a) | set(count_b)) / (n * n)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> dict[str, Any]:
    n = len(a)
    return {
        "items": n,
        "agree_rate": round(sum(x == y for x, y in zip(a, b, strict=True)) / n, 4) if n else None,
        "kappa": cohen_kappa(a, b),
    }
