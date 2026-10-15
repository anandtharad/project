"""Experiment metadata for evaluation runs (spec §19.3, Phase 6).

Everything that decides a run's behaviour is recorded with its results, so
two runs can be compared and a failure traced to a component, model, prompt,
policy or retrieval version:

- model/provider and generation parameters;
- prompt ids, versions and checksums, including the policy fragments;
- decision provider, model, question versions and calibration key;
- safety and knowledge settings, the knowledge manifest checksum and the
  embedding model (no reranker is used);
- the dataset file and its checksum, the code revision, and the time.

Keys and secrets are never included (only whether a key is set is implied by
the provider that resolved). Known limitations are supplied by the caller.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.decisions.questions import LOCK_FILE as QUESTION_LOCK
from app.decisions.questions import QUESTIONS
from app.llm.providers.registry import resolve_provider
from app.prompting import PromptRegistry
from app.rag.manifest import DEFAULT_MANIFEST

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def code_revision() -> dict[str, Any]:
    """The git commit and whether the working tree had changes (None when git
    is unavailable, e.g. in an exported copy)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def _prompt(registry: PromptRegistry, prompt_id: str, version: str) -> dict[str, Any]:
    try:
        prompt = registry.get(prompt_id, version)
    except Exception as exc:  # recorded, not raised: metadata must not fail a run
        return {"id": prompt_id, "version": version, "error": type(exc).__name__}
    return {"id": prompt.id, "version": prompt.version, "checksum": prompt.checksum}


def experiment_metadata(
    settings: Settings,
    *,
    suite: str,
    dataset: Path | None = None,
    limitations: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = PromptRegistry()
    resolved = resolve_provider(settings)
    return {
        "suite": suite,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_revision(),
        "python": platform.python_version(),
        "generation": {
            "provider": resolved.preset.name,
            "model": resolved.model,
            "effort": resolved.effort,
            "temperature": settings.generation_temperature,
            "max_output_tokens": resolved.max_output_tokens,
            "max_context_tokens": resolved.max_context_tokens,
        },
        "prompts": {
            "companion": _prompt(registry, "companion_system", settings.companion_prompt_version),
            "state": _prompt(registry, "state_extraction", settings.state_prompt_version),
            "memory": _prompt(registry, "memory_extraction", settings.memory_prompt_version),
            "decision_llm": _prompt(registry, "decision_llm", settings.decision_prompt_version),
            "safety_policy": _prompt(registry, "safety_policy", settings.safety_policy_version),
            "safety_output": _prompt(
                registry, "safety_output", settings.safety_output_policy_version
            ),
        },
        "decisions": {
            "provider": settings.decision_provider,
            "llm_model": (
                settings.decision_llm_model if settings.decision_provider == "llm" else None
            ),
            "questions": {q.id: q.version for q in QUESTIONS},
            "questions_lock_sha256": file_sha256(QUESTION_LOCK),
        },
        "state_extraction": settings.state_extraction_enabled,
        "memory": {
            "enabled": settings.memory_enabled,
            "min_semantic": settings.memory_min_semantic,
            "max_retrieved": settings.memory_max_retrieved,
        },
        "embeddings": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
        },
        "knowledge": {
            "enabled": settings.knowledge_enabled,
            "max_results": settings.knowledge_max_results,
            "min_semantic": settings.knowledge_min_semantic,
            "min_lexical": settings.knowledge_min_lexical,
            "max_per_document": settings.knowledge_max_per_document,
            "chunking": (
                f"{settings.knowledge_chunk_max_words}/{settings.knowledge_chunk_overlap_words}/"
                f"{settings.knowledge_chunk_min_words}"
            ),
            "manifest_sha256": file_sha256(DEFAULT_MANIFEST),
            "reranker": None,
        },
        "safety": {
            "enabled": settings.safety_enabled,
            "classifier_provider": settings.safety_classifier_provider,
            "classifier_model": settings.safety_classifier_model,
            "region": settings.safety_region,
            "output_llm_check": settings.safety_output_llm_check,
        },
        "dataset": (
            {
                "path": str(dataset.relative_to(ROOT) if dataset.is_relative_to(ROOT) else dataset),
                "sha256": file_sha256(dataset),
            }
            if dataset is not None
            else None
        ),
        "limitations": list(limitations),
        **(extra or {}),
    }


def diff(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[tuple[str, Any, Any]]:
    """Differences between two metadata records, as (dotted key, a, b);
    timestamps are ignored."""
    changes: list[tuple[str, Any, Any]] = []
    for key in sorted(set(a) | set(b)):
        if key == "created_at":
            continue
        left, right = a.get(key), b.get(key)
        path = f"{prefix}{key}"
        if isinstance(left, dict) and isinstance(right, dict):
            changes += diff(left, right, f"{path}.")
        elif left != right:
            changes.append((path, left, right))
    return changes
