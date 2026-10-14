"""Evaluate decision questions on a labelled set (spec §7.8).

For each conversation the decision state is built exactly as in a live turn
(`build_decision_state`), every question is asked in one request, and each
answer is validated against the question. Metrics per question:

- choice: accuracy and macro-F1 against the gold label, the majority-class
  baseline on the same split, per-label recall, a confusion matrix,
  expected calibration error (10 bins, probability of the chosen label), and
  a confidence-threshold sweep (coverage and accuracy of the answers acted
  on) used to choose a threshold on dev data.
- noul: accuracy at 0.5, Brier score, AUROC and the base rate.

Thresholds are chosen on `dev` only: the lowest confidence at which the
answers acted on reach the target accuracy with enough coverage. They are
then checked, unchanged, on the held-out `test` split, and only with results
from the same calibration identity (`run_identity`: provider, model, decision
prompt, effort and output limit, question checksums).

A run can be paced for a provider's free tier (`Pacer`), stops instead of
recording a failure when a rate limit or quota is hit (`RunStopped`: a rate
limit says nothing about the decision model), and can be continued later from
cached answers (`load_cache` / `merge_predictions`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.decisions.backend import DecisionBackendError
from app.decisions.llm_backend import LLMDecisionBackend
from app.decisions.schemas import FallbackReason, QuestionSpec
from app.decisions.service import DecisionBackend, build_decision_state, validate_answer
from app.llm.schemas import ChatMessage, ChatRole

THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(20))
TARGET_ACCURACY = 0.80
MIN_ACTED = 10
MIN_COVERAGE = 0.15

# Rate limits: how often one example is retried, and the backoff when the
# provider gives no retry-after.
RATE_LIMIT_ATTEMPTS = 5
RATE_LIMIT_BACKOFF_SECONDS = 10.0

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class Example:
    id: str
    conversation: list[ChatMessage]
    labels: dict[str, Any]
    difficulty: str
    family: str


@dataclass
class Prediction:
    example_id: str
    answers: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float | None = None
    model_id: str | None = None


def load_examples(path: Path) -> list[Example]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(
            Example(
                id=row["id"],
                conversation=[
                    ChatMessage(role=ChatRole(m["role"]), content=m["content"])
                    for m in row["conversation"]
                ],
                labels=row["labels"],
                difficulty=row.get("difficulty", "clear"),
                family=row.get("family", ""),
            )
        )
    return examples


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunStopped(Exception):
    """A rate limit or quota stopped the run before every example was answered.

    No failure is recorded for the example that hit it: on a free tier a 429
    is a property of the account, not of the decision model, and counting it
    would silently lower accuracy and coverage. Answers obtained earlier were
    already passed to `on_prediction`, so the run can be resumed.
    """

    def __init__(self, reason: str, answered: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.answered = answered


class Pacer:
    """Spaces requests for a provider's free tier (Groq's gpt-oss-20b: 30
    requests and 8K tokens per minute).

    `pause_seconds` is the minimum gap between one request's end and the next
    request. `tokens_per_minute` keeps the tokens used in the last 60 seconds,
    plus the expected size of the next request (the largest seen so far),
    within that budget.
    """

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        *,
        pause_seconds: float = 0.0,
        tokens_per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._pause = pause_seconds
        self._budget = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._used: deque[tuple[float, int]] = deque()
        self._last_end: float | None = None
        self._expected = 0

    async def wait(self) -> None:
        if self._last_end is not None and self._pause > 0:
            delay = self._last_end + self._pause - self._clock()
            if delay > 0:
                await self._sleep(delay)
        if not self._budget:
            return
        while self._used:
            now = self._clock()
            while self._used and self._used[0][0] <= now - self.WINDOW_SECONDS:
                self._used.popleft()
            if not self._used or sum(t for _, t in self._used) + self._expected <= self._budget:
                return
            await self._sleep(self._used[0][0] + self.WINDOW_SECONDS - now)

    def record(self, tokens: int | None) -> None:
        """Call after each request, with the tokens it used when known."""
        self._last_end = self._clock()
        if tokens:
            self._used.append((self._last_end, tokens))
            self._expected = max(self._expected, tokens)


async def predict(
    backend: DecisionBackend,
    questions: Sequence[QuestionSpec],
    examples: Sequence[Example],
    *,
    progress: bool = False,
    pacer: Pacer | None = None,
    on_prediction: Callable[[Prediction], None] | None = None,
    max_wait_seconds: float = 120.0,
    sleep: Sleep = asyncio.sleep,
) -> list[Prediction]:
    """Ask every example once. Timeouts, outages and invalid responses are
    recorded as failures, as a live turn would fall back on them. A rate limit
    is waited out (the provider's retry-after, or a backoff) when the wait is at
    most `max_wait_seconds`; a longer wait, a used-up quota or repeated limits
    raise `RunStopped`."""
    predictions: list[Prediction] = []
    for index, example in enumerate(examples, 1):
        state = build_decision_state(example.conversation)
        prediction = Prediction(example_id=example.id)
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            if pacer is not None:
                await pacer.wait()
            started = time.perf_counter()
            try:
                raw = await backend.ask(state, list(questions))
            except DecisionBackendError as exc:
                if pacer is not None:
                    pacer.record(None)
                if exc.reason is FallbackReason.RATE_LIMITED:
                    wait = exc.retry_after_seconds
                    if wait is None:
                        wait = RATE_LIMIT_BACKOFF_SECONDS * 2**attempt
                    if exc.quota_exhausted:
                        stop = "the provider reports its quota is used up"
                    elif wait > max_wait_seconds:
                        stop = f"the provider asks to wait {wait:.0f} s (more than --max-wait)"
                    elif attempt == RATE_LIMIT_ATTEMPTS - 1:
                        stop = f"still rate limited after {RATE_LIMIT_ATTEMPTS} attempts"
                    else:
                        print(f"  {example.id}: rate limited, waiting {wait:.0f} s", flush=True)
                        await sleep(wait)
                        continue
                    raise RunStopped(stop, answered=len(predictions)) from exc
                prediction.error = exc.reason.value
            else:
                if pacer is not None:
                    tokens = (raw.input_tokens or 0) + (raw.output_tokens or 0)
                    pacer.record(tokens or None)
                prediction.model_id = raw.model_id
                for question in questions:
                    answer = raw.answers.get(question.id)
                    prediction.answers[question.id] = (
                        answer if answer is not None and validate_answer(question, answer) else None
                    )
            prediction.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            break
        predictions.append(prediction)
        if on_prediction is not None:
            on_prediction(prediction)
        if progress and index % 25 == 0:
            print(f"  {index}/{len(examples)}", flush=True)
    return predictions


# --- Run identity, caching and result files -----------------------------------

# What thresholds depend on. A dev result whose identity differs from the
# current run's cannot supply its thresholds (spec §7.8 item 8, §19.3).
IDENTITY_FIELDS = ("provider", "requested_model", "decision_llm", "decision_prompt", "questions")


def run_identity(backend: DecisionBackend, questions: Sequence[QuestionSpec]) -> dict[str, Any]:
    """Everything a result's numbers depend on besides the dataset."""
    identity: dict[str, Any] = {
        "provider": backend.provider,
        "requested_model": backend.model,
        "decision_llm": None,
        "decision_prompt": None,
        "questions": {q.key: q.checksum for q in questions},
    }
    if isinstance(backend, LLMDecisionBackend):
        identity["decision_llm"] = {
            "provider": backend.llm_provider,
            "effort": backend.effort,
            "max_output_tokens": backend.max_output_tokens,
        }
        identity["decision_prompt"] = {
            "key": backend.prompt.key,
            "checksum": backend.prompt.checksum,
            # The rendered system prompt and response schema, so a change in
            # how questions are laid out is caught too.
            "request_sha256": backend.request_checksum(questions),
        }
    return identity


def thresholds_source_problems(source: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    """Why a result file cannot supply thresholds for a run with `identity`
    (empty when it can). Files written before a field existed read it as None,
    which matches a jev/laya run."""
    problems = []
    if source.get("split") != "dev":
        problems.append(f"it is a {source.get('split')!r} result; thresholds come from dev only")
    for name in IDENTITY_FIELDS:
        if source.get(name) != identity.get(name):
            problems.append(
                f"{name} differs: {source.get(name)!r} here, {identity.get(name)!r} now"
            )
    return problems


def selected_thresholds(source: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(m["selected_threshold"])
        for key, m in source["metrics"]["questions"].items()
        if m.get("selected_threshold") is not None
    }


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(part: str) -> str:
    """A file-name part on every OS: "/" (a directory), ":" and other
    characters Windows refuses become "_"."""
    return _UNSAFE_NAME.sub("_", part).strip("._") or "_"


def run_label(identity: Mapping[str, Any]) -> str:
    """provider[-llm provider]-model[-prompt], safe for a file name."""
    parts = [str(identity["provider"])]
    if identity.get("decision_llm"):
        parts.append(str(identity["decision_llm"]["provider"]))
    parts.append(str(identity["requested_model"]))
    if identity.get("decision_prompt"):
        parts.append(str(identity["decision_prompt"]["key"]))
    return "-".join(safe_name(p) for p in parts)


def result_file_name(identity: Mapping[str, Any], split: str, stamp: str) -> str:
    return f"decisions-{run_label(identity)}-{safe_name(split)}-{stamp}.json"


def cache_identity(
    identity: Mapping[str, Any], *, dataset_sha256: str, split: str
) -> dict[str, Any]:
    return {**identity, "dataset_sha256": dataset_sha256, "split": split}


def cache_file_name(cache_id: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(cache_id, sort_keys=True).encode("utf-8")).hexdigest()
    return f"decisions-{run_label(cache_id)}-{safe_name(str(cache_id['split']))}-{digest[:12]}.json"


def load_cache(path: Path, cache_id: Mapping[str, Any]) -> dict[str, Prediction]:
    """Cached answers for exactly this identity, dataset and split. A cache
    written for anything else is refused, never merged."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("identity") != dict(cache_id):
        raise ValueError(f"{path} was written for a different run; move it away or delete it")
    return {key: Prediction(**value) for key, value in data["predictions"].items()}


def save_cache(
    path: Path, cache_id: Mapping[str, Any], predictions: Mapping[str, Prediction]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": dict(cache_id),
        "predictions": {key: dataclasses.asdict(p) for key, p in predictions.items()},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    temporary.replace(path)  # never leave a half-written cache behind


def merge_predictions(
    examples: Sequence[Example], cached: Mapping[str, Prediction], new: Sequence[Prediction]
) -> tuple[list[Prediction], list[str]]:
    """Predictions in example order (new answers replace cached ones), and the
    ids of examples that have none yet."""
    by_id = {**cached, **{p.example_id: p for p in new}}
    merged = [by_id[e.id] for e in examples if e.id in by_id]
    missing = [e.id for e in examples if e.id not in by_id]
    return merged, missing


# --- Metrics -----------------------------------------------------------------


def _ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float | None:
    if not confidences:
        return None
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            i for i, c in enumerate(confidences) if (low < c <= high) or (b == 0 and c == 0.0)
        ]
        if members:
            accuracy = sum(correct[i] for i in members) / len(members)
            mean_conf = sum(confidences[i] for i in members) / len(members)
            total += len(members) / len(confidences) * abs(accuracy - mean_conf)
    return round(total, 4)


def _auroc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    pos = [s for s, p in zip(scores, positives, strict=True) if p]
    neg = [s for s, p in zip(scores, positives, strict=True) if not p]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def threshold_sweep(confidences: Sequence[float], correct: Sequence[bool]) -> list[dict[str, Any]]:
    rows = []
    n = len(confidences)
    for t in THRESHOLD_GRID:
        acted = [ok for c, ok in zip(confidences, correct, strict=True) if c >= t]
        rows.append(
            {
                "threshold": t,
                "acted": len(acted),
                "coverage": round(len(acted) / n, 4) if n else 0.0,
                "accuracy": round(sum(acted) / len(acted), 4) if acted else None,
            }
        )
    return rows


def select_threshold(sweep: Sequence[Mapping[str, Any]]) -> float | None:
    """Lowest threshold whose acted-on answers reach the target accuracy with
    enough examples and coverage; None when no threshold qualifies."""
    for row in sweep:
        if (
            row["accuracy"] is not None
            and row["accuracy"] >= TARGET_ACCURACY
            and row["acted"] >= MIN_ACTED
            and row["coverage"] >= MIN_COVERAGE
        ):
            return float(row["threshold"])
    return None


def at_threshold(
    confidences: Sequence[float], correct: Sequence[bool], threshold: float
) -> dict[str, Any]:
    acted = [ok for c, ok in zip(confidences, correct, strict=True) if c >= threshold]
    return {
        "threshold": threshold,
        "acted": len(acted),
        "coverage": round(len(acted) / len(confidences), 4) if confidences else 0.0,
        "accuracy": round(sum(acted) / len(acted), 4) if acted else None,
    }


def choice_metrics(
    question: QuestionSpec,
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    threshold: float | None = None,
) -> dict[str, Any]:
    gold_all = [e.labels.get(question.id) for e in examples]
    scored = [(g, p) for g, p in zip(gold_all, predictions, strict=True) if g is not None]
    gold = [g for g, _ in scored]
    answers = [p.answers.get(question.id) for _, p in scored]
    predicted = [a["choice"] if a else None for a in answers]
    correct = [p == g for p, g in zip(predicted, gold, strict=True)]
    n = len(gold)
    majority_label, majority_count = Counter(gold).most_common(1)[0] if gold else (None, 0)
    labels = list(question.options)
    f1s = []
    per_label: dict[str, Any] = {}
    for label in labels:
        tp = sum(p == label and g == label for p, g in zip(predicted, gold, strict=True))
        fp = sum(p == label and g != label for p, g in zip(predicted, gold, strict=True))
        fn = sum(p != label and g == label for p, g in zip(predicted, gold, strict=True))
        support = tp + fn
        if support:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            per_label[label] = {
                "support": support,
                "recall": round(recall, 3),
                "predicted": tp + fp,
            }
    confusion: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, predicted, strict=True):
        row = confusion.setdefault(g, {})
        row[str(p)] = row.get(str(p), 0) + 1
    valid = [(a, ok) for a, ok in zip(answers, correct, strict=True) if a is not None]
    confidences = [float(a["confidence"]) for a, _ in valid]
    valid_correct = [ok for _, ok in valid]
    top_probability = [float(a["probabilities"][a["choice"]]) for a, _ in valid]
    # Thresholds are swept over every scored example: a failed request or an
    # invalid answer falls back in a live turn, so it is never acted on and
    # must count in the coverage denominator (-1 is below every threshold).
    all_confidences = [float(a["confidence"]) if a is not None else -1.0 for a in answers]
    sweep = threshold_sweep(all_confidences, correct)
    result: dict[str, Any] = {
        "n": n,
        "valid_answers": len(valid),
        "accuracy": round(sum(correct) / n, 4) if n else None,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "majority_baseline": {
            "label": majority_label,
            "accuracy": round(majority_count / n, 4) if n else None,
        },
        "ece_top_probability": _ece(top_probability, valid_correct),
        "mean_confidence": round(statistics.fmean(confidences), 4) if confidences else None,
        "predicted_distribution": dict(Counter(str(p) for p in predicted)),
        "per_label": per_label,
        "confusion": confusion,
        "threshold_sweep": sweep,
        "selected_threshold": select_threshold(sweep),
    }
    if threshold is not None:
        result["at_given_threshold"] = at_threshold(all_confidences, correct, threshold)
    return result


def noul_metrics(
    question: QuestionSpec, examples: Sequence[Example], predictions: Sequence[Prediction]
) -> dict[str, Any]:
    pairs = [
        (bool(e.labels[question.id]), p.answers.get(question.id))
        for e, p in zip(examples, predictions, strict=True)
        if e.labels.get(question.id) is not None
    ]
    valid = [(g, float(a["noul"])) for g, a in pairs if a is not None]
    if not valid:
        return {"n": len(pairs), "valid_answers": 0}
    gold = [g for g, _ in valid]
    probabilities = [p for _, p in valid]
    return {
        "n": len(pairs),
        "valid_answers": len(valid),
        "base_rate": round(sum(gold) / len(gold), 4),
        "accuracy_at_0_5": round(sum((p >= 0.5) == g for g, p in valid) / len(valid), 4),
        "majority_baseline_accuracy": round(max(sum(gold), len(gold) - sum(gold)) / len(gold), 4),
        "brier": round(sum((p - g) ** 2 for g, p in valid) / len(valid), 4),
        "auroc": _auroc(probabilities, gold),
        "mean_probability": round(statistics.fmean(probabilities), 4),
        "probability_range": [round(min(probabilities), 4), round(max(probabilities), 4)],
    }


def summarize(
    questions: Sequence[QuestionSpec],
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    latencies = sorted(p.latency_ms for p in predictions if p.latency_ms is not None)
    metrics: dict[str, Any] = {}
    for question in questions:
        if question.kind == "choice":
            given = (thresholds or {}).get(question.key)
            metrics[question.key] = choice_metrics(question, examples, predictions, given)
        elif question.kind == "noul":
            metrics[question.key] = noul_metrics(question, examples, predictions)
    safety = [
        p.answers.get("conversation_mode")
        for e, p in zip(examples, predictions, strict=True)
        if e.labels.get("conversation_mode") == "safety"
    ]
    return {
        "examples": len(examples),
        "failed_requests": dict(Counter(p.error for p in predictions if p.error)),
        "latency_ms": {
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None,
        },
        "safety_mode_recall": {
            "gold_safety": len(safety),
            "predicted_safety": sum(1 for a in safety if a and a["choice"] == "safety"),
        },
        "questions": metrics,
    }
