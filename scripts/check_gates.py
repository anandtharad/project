"""Check the release gates in eval/gates.toml against the latest results.

    uv run python scripts/check_gates.py            # table; exit 1 if a mandatory gate fails
    uv run python scripts/check_gates.py --strict   # exit 1 if any gate fails

A gate with no matching result file is reported as "no result", which fails
it when mandatory. Paths are dotted, with [n] for list items:
"runs[0].fixed.metrics.hit_rate".
"""

from __future__ import annotations

import argparse
import json
import operator
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GATES = ROOT / "eval" / "gates.toml"
RESULTS = ROOT / "eval" / "results"
OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}
_STEP = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def lookup(data: Any, path: str) -> Any:
    value = data
    for key, index in _STEP.findall(path):
        if index:
            if not isinstance(value, list) or int(index) >= len(value):
                return None
            value = value[int(index)]
        else:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
    return value


def latest(pattern: str, require: Mapping[str, Any]) -> tuple[Path, Any] | None:
    # Result names end in a UTC timestamp, so name order is time order.
    for path in sorted(RESULTS.glob(pattern), reverse=True):
        if path.name.endswith((".packets.json", ".replies.json")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if all(lookup(data, key) == value for key, value in require.items()):
            return path, data
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strict", action="store_true", help="fail on any gate")
    args = parser.parse_args()
    config = tomllib.loads(GATES.read_text(encoding="utf-8"))
    failed_mandatory = failed_any = 0
    print(f"{'gate':<74} {'value':>9} {'threshold':>12} {'baseline':>9}  result")
    for gate in config["gates"]:
        found = latest(gate["results"], gate.get("require", {}))
        mandatory = bool(gate.get("mandatory", False))
        marker = "*" if mandatory else " "
        if found is None:
            status, shown = "no result", "-"
            ok = False
        else:
            path, data = found
            value = lookup(data, gate["path"])
            shown = f"{value:.4g}" if isinstance(value, int | float) else str(value)
            ok = isinstance(value, int | float) and OPS[gate["op"]](value, gate["threshold"])
            status = f"pass ({path.name})" if ok else f"FAIL ({path.name})"
        if not ok:
            failed_any += 1
            failed_mandatory += mandatory
        threshold = f"{gate['op']} {gate['threshold']}"
        print(
            f"{marker}{gate['name']:<73} {shown:>9} {threshold:>12} {gate['baseline']:>9}  {status}"
        )
    print(f"\n* mandatory. Failed: {failed_mandatory} mandatory, {failed_any} in all.")
    if failed_mandatory or (args.strict and failed_any):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
