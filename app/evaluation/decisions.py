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
then checked, unchanged, on the held-out `test` split.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.decisions.backend import DecisionBackendError
from app.decisions.schemas import QuestionSpec
from app.decisions.service import DecisionBackend, build_decision_state, validate_answer
from app.llm.schemas import ChatMessage, ChatRole

THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(20))
TARGET_ACCURACY = 0.80
MIN_ACTED = 10
MIN_COVERAGE = 0.15


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


async def predict(
    backend: DecisionBackend,
    questions: Sequence[QuestionSpec],
    examples: Sequence[Example],
    *,
    progress: bool = False,
) -> list[Prediction]:
    predictions = []
    for index, example in enumerate(examples, 1):
        prediction = Prediction(example_id=example.id)
        started = time.perf_counter()
        try:
            raw = await backend.ask(build_decision_state(example.conversation), list(questions))
        except DecisionBackendError as exc:
            prediction.error = exc.reason.value
        else:
            prediction.model_id = raw.model_id
            for question in questions:
                answer = raw.answers.get(question.id)
                prediction.answers[question.id] = (
                    answer if answer is not None and validate_answer(question, answer) else None
                )
        prediction.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        predictions.append(prediction)
        if progress and index % 25 == 0:
            print(f"  {index}/{len(examples)}", flush=True)
    return predictions


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
