from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.prompting import (
    LOCK_FILE_NAME,
    PROMPTS_DIR,
    PromptError,
    PromptRegistry,
    discover_prompts,
    load_lock,
)


def test_every_prompt_is_locked_with_matching_checksum() -> None:
    lock = load_lock()
    prompts = discover_prompts()
    assert prompts, "no prompts found"
    for prompt in prompts:
        assert lock.get(prompt.key) == prompt.checksum, (
            f"{prompt.key} changed after release or is unlocked; "
            "create a new version and run scripts/lock_prompts.py"
        )


def test_prompt_metadata_is_complete() -> None:
    for prompt in discover_prompts():
        assert prompt.intended_use.strip()
        assert prompt.changelog
        assert prompt.changelog[-1].startswith(prompt.version)


def test_companion_prompt_covers_required_behaviors() -> None:
    text = PromptRegistry().get("companion_system", "1.0.0").render()
    for phrase in (
        "You are an AI, not a person",
        "not a therapist",
        "Respond to what the person actually said",
        "don't slip it in",
        "Many good replies end without a question",
        "Offer interpretations tentatively",
        "Don't diagnose",
        "If you're unsure about a fact, say so",
        "don't present yourself as their only support",
        "local emergency services or a crisis line",
        "do not change them",
    ):
        assert phrase in text, phrase


def copy_prompts(tmp_path: Path) -> Path:
    root = tmp_path / "prompts"
    shutil.copytree(PROMPTS_DIR, root)
    return root


def test_edited_released_prompt_is_rejected(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    path = root / "companion_system" / "1.0.0.toml"
    path.write_text(path.read_text().replace('template = """', 'template = """Be verbose. ', 1))
    with pytest.raises(PromptError, match="immutable"):
        PromptRegistry(root).get("companion_system", "1.0.0")


def test_unlocked_prompt_is_rejected(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (root / LOCK_FILE_NAME).write_text(json.dumps({}))
    with pytest.raises(PromptError, match="not recorded"):
        PromptRegistry(root).get("companion_system", "1.0.0")


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(PromptError, match="Unknown prompt"):
        PromptRegistry().get("companion_system", "9.9.9")


def test_variables_must_match_declaration(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "1.0.0.toml").write_text(
        'id = "demo"\nversion = "1.0.0"\nintended_use = "test"\n'
        'variables = []\nchangelog = ["1.0.0: test"]\ntemplate = "Hello $name"\n'
    )
    with pytest.raises(PromptError, match="declared variables"):
        discover_prompts(root)


def test_render_requires_exact_variables(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "1.0.0.toml").write_text(
        'id = "demo"\nversion = "1.0.0"\nintended_use = "test"\n'
        'variables = ["name"]\nchangelog = ["1.0.0: test"]\ntemplate = "Hello $name"\n'
    )
    (prompt,) = discover_prompts(root)
    assert prompt.render(name="Sam") == "Hello Sam"
    with pytest.raises(PromptError):
        prompt.render()
    with pytest.raises(PromptError):
        prompt.render(name="Sam", extra="x")
