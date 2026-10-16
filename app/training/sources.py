"""The training-source audit registry (training/sources.toml, spec §17.2).

Each source records what §17.2 asks: dataset ID, URL, licence, permitted
use, split, origin and synthetic status, quality annotations, and an
approval. A build reads only approved sources; an approval names who
approved it and when.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE = ROOT / "training" / "sources.toml"


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    approved_by: str = ""
    approved_on: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _signed(self) -> Approval:
        if self.approved and not (self.approved_by.strip() and self.approved_on.strip()):
            raise ValueError("an approval needs approved_by and approved_on")
        return self


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    url: str
    file: str
    file_url: str
    sha256: str
    licence: str
    permitted_use: str
    citation: str
    synthetic: bool
    origin: str
    splits: str
    quality_annotations: str
    licence_checked_on: str
    approval: Approval


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources_version: int
    sources: list[Source]

    def get(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)


def load_registry(path: Path = SOURCES_FILE) -> Registry:
    registry = Registry.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    ids = [s.id for s in registry.sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate source ids in {path}")
    return registry
