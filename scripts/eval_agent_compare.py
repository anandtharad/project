"""Compare a golden run with AGENT_MODE=on against one without (docs/agent.md).

    uv run python scripts/eval_golden.py --mode live --split dev --pause 20
    uv run python scripts/eval_golden.py --mode live --split dev --pause 20 --agent
    uv run python scripts/eval_agent_compare.py --mode live --split dev

By default the latest golden-<mode>-<split>-*.json and
golden-<mode>-agent-<split>-*.json in eval/results are compared (or pass
--base and --agent paths). Writes eval/results/agent-compare-<mode>-<split>-<time>.json
and prints the headline numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation.agent_compare import compare  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"


def latest(pattern: str) -> Path:
    found = sorted(p for p in RESULTS_DIR.glob(pattern) if not p.name.endswith(".replies.json"))
    if not found:
        raise SystemExit(f"no result matching eval/results/{pattern}; run eval_golden.py first")
    return found[-1]


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replies = path.with_suffix(".replies.json")
    return (
        json.loads(path.read_text(encoding="utf-8")),
        json.loads(replies.read_text(encoding="utf-8")) if replies.exists() else {},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["deterministic", "live"], default="live")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--profile", choices=["lean", "full"], default="lean")
    parser.add_argument("--base", type=Path, help="result file of the run without the agent")
    parser.add_argument("--agent", type=Path, help="result file of the run with the agent")
    args = parser.parse_args()
    base_path = args.base or latest(f"golden-{args.mode}-{args.split}-*.json")
    agent_path = args.agent or latest(
        f"golden-{args.mode}-agent-{args.profile}-{args.split}-*.json"
    )
    base, base_replies = load(base_path)
    agent, agent_replies = load(agent_path)
    report = {
        "eval": "agent-compare",
        "base": base_path.name,
        "agent": agent_path.name,
        **compare(base, agent, base_replies, agent_replies),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"agent-compare-{args.profile}-{args.mode}-{args.split}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'':28}{'base':>12}{'agent':>12}")
    for name, pair in report["rates"].items():
        print(f"{name:28}{pair['base']!s:>12}{pair['agent']!s:>12}")
    k = report["knowledge_decisions"]
    print(
        f"{'knowledge agreement':28}{k['base']['agreement']!s:>12}{k['agent']['agreement']!s:>12}"
    )
    for key in ("seconds", "tokens", "input_tokens", "state_tokens"):
        b, a = report["cost"]["base"][key], report["cost"]["agent"][key]
        print(f"{key + ' (median)':28}{b['median']!s:>12}{a['median']!s:>12}")
    behaviour = report["agent"]
    print(json.dumps({k: behaviour.get(k) for k in ("statuses", "strategies", "tools")}))
    print(
        f"changed: {len(report['changed']['better'])} better, "
        f"{len(report['changed']['worse'])} worse"
    )
    print(f"written: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
