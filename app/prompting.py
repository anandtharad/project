"""Versioned prompt registry.

Prompts live in `prompts/<prompt_id>/<version>.toml`. Each file declares its
id, version, intended use, expected variables, changelog, and template.
`prompts/prompts.lock.json` pins the SHA-256 of every released template, so a
released prompt cannot be edited in place: changing behavior requires a new
version (run `scripts/lock_prompts.py` to record it).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
LOCK_FILE_NAME = "prompts.lock.json"


class PromptError(Exception):
    pass


def template_checksum(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    intended_use: str
    variables: tuple[str, ...]
    changelog: tuple[str, ...]
    template: str

    @property
    def checksum(self) -> str:
        return template_checksum(self.template)

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    def render(self, **values: str) -> str:
        expected = set(self.variables)
        provided = set(values)
        if provided != expected:
            raise PromptError(
                f"{self.key}: expected variables {sorted(expected)}, got {sorted(provided)}"
            )
        return Template(self.template).substitute(values).strip()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    # Windows tools may save with a UTF-8 BOM, or as UTF-16 (PowerShell 5.1 redirection).
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def load_prompt_file(path: Path) -> PromptTemplate:
    try:
        data = tomllib.loads(_read_text(path))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PromptError(f"{path.name}: {exc}") from exc
    try:
        prompt = PromptTemplate(
            id=str(data["id"]),
            version=str(data["version"]),
            intended_use=str(data["intended_use"]),
            variables=tuple(str(v) for v in data["variables"]),
            changelog=tuple(str(c) for c in data["changelog"]),
            template=str(data["template"]),
        )
    except KeyError as exc:
        raise PromptError(f"{path}: missing field {exc}") from exc
    if path.parent.name != prompt.id or path.stem != prompt.version:
        raise PromptError(f"{path}: id/version do not match the file location")
    if not prompt.changelog:
        raise PromptError(f"{path}: changelog must not be empty")
    template = Template(prompt.template)
    if not template.is_valid():
        raise PromptError(f"{path}: template has invalid placeholders")
    if set(template.get_identifiers()) != set(prompt.variables):
        raise PromptError(f"{path}: declared variables do not match template placeholders")
    return prompt


def load_lock(root: Path = PROMPTS_DIR) -> dict[str, str]:
    lock_path = root / LOCK_FILE_NAME
    if not lock_path.exists():
        return {}
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PromptError(f"{lock_path}: expected a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def discover_prompts(root: Path = PROMPTS_DIR) -> list[PromptTemplate]:
    return [load_prompt_file(p) for p in sorted(root.glob("*/*.toml"))]


class PromptRegistry:
    def __init__(self, root: Path = PROMPTS_DIR) -> None:
        self._root = root
        self._lock = load_lock(root)

    def get(self, prompt_id: str, version: str) -> PromptTemplate:
        path = self._root / prompt_id / f"{version}.toml"
        if not path.exists():
            raise PromptError(f"Unknown prompt {prompt_id}@{version}")
        prompt = load_prompt_file(path)
        locked = self._lock.get(prompt.key)
        if locked is None:
            raise PromptError(f"{prompt.key} is not recorded in {LOCK_FILE_NAME}")
        if locked != prompt.checksum:
            raise PromptError(
                f"{prompt.key} differs from its locked checksum; "
                "released prompts are immutable - create a new version instead"
            )
        return prompt
