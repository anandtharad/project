"""Compare the context builder's token estimate with the input tokens the
provider reported, per model, from stored replies.

    uv run python scripts/token_report.py            # DATABASE_URL from .env

The context budget (app/context/builder.py) estimates tokens from
characters (4 per token) because no single tokenizer fits every provider.
Every reply stores both the estimate (generation_metadata.context
.estimated_tokens) and the provider's count (generation_metadata.usage
.input_tokens), so the estimate can be checked on real turns: a ratio below
1 means the estimate undercounts and the budget is too generous. Reads
numbers only, never message text.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.config import Settings  # noqa: E402

QUERY = text(
    "SELECT provider, model_id,"
    " (generation_metadata -> 'context' ->> 'estimated_tokens')::int AS estimated,"
    " (generation_metadata -> 'usage' ->> 'input_tokens')::int AS actual"
    " FROM messages WHERE role = 'assistant' AND generation_metadata IS NOT NULL"
)


def summarize(rows: list[tuple[str, str, int | None, int | None]]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for provider, model, estimated, actual in rows:
        if estimated and actual:
            by_model[f"{provider}:{model}"].append(estimated / actual)
    out = {}
    for model, ratios in sorted(by_model.items()):
        ratios.sort()
        out[model] = {
            "replies": len(ratios),
            "median_estimate_over_actual": round(statistics.median(ratios), 3),
            "lowest": round(ratios[0], 3),
            "p10": round(ratios[max(0, len(ratios) // 10 - 1)], 3),
        }
    return out


async def main() -> int:
    engine = create_async_engine(Settings().database_url)
    try:
        async with engine.connect() as conn:
            rows = [tuple(r) for r in (await conn.execute(QUERY)).all()]
    finally:
        await engine.dispose()
    report = summarize(rows)
    if not report:
        print("No stored reply has both an estimate and a provider token count yet.")
        return 0
    for model, stats in report.items():
        print(model, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
