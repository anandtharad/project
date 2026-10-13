"""User memory controls (spec §10.7, §14.3).

All routes act only on the authenticated user's own memories; another
person's memory ID returns the same 404 as a missing one. Deleting removes
the row and its embedding (stored in the same row).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict

from app.api.auth import AuthenticatedUser, authenticate
from app.errors import memory_not_found
from app.memory.policy import MemorySettings
from app.memory.schemas import Memory, MemoryStatus
from app.memory.store import MemoryStore

router = APIRouter(prefix="/v1/users/me", tags=["memories"])

CurrentUser = Annotated[AuthenticatedUser, Depends(authenticate)]
StatusFilter = Literal["active", "pending_confirmation", "superseded", "all"]


class MemoryOut(BaseModel):
    memory_id: uuid.UUID
    memory_type: str
    content: str
    status: str
    importance: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    source_conversation_id: uuid.UUID | None
    supersedes_id: uuid.UUID | None

    @classmethod
    def of(cls, m: Memory) -> MemoryOut:
        return cls(
            memory_id=m.id,
            memory_type=m.memory_type.value,
            content=m.content,
            status=m.status.value,
            importance=m.importance,
            created_at=m.created_at,
            updated_at=m.updated_at,
            expires_at=m.expires_at,
            source_conversation_id=m.source_conversation_id,
            supersedes_id=m.supersedes_id,
        )


class MemoryList(BaseModel):
    memories: list[MemoryOut]


class DeletedCount(BaseModel):
    deleted: int


class MemorySettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    memory_enabled: bool
    allow_sensitive: bool


def _store(request: Request) -> MemoryStore:
    store: MemoryStore = request.app.state.memory_store
    return store


@router.get("/memories")
async def list_memories(
    request: Request,
    user: CurrentUser,
    status_filter: Annotated[StatusFilter | None, Query(alias="status")] = None,
) -> MemoryList:
    if status_filter is None:
        statuses = [MemoryStatus.ACTIVE, MemoryStatus.PENDING_CONFIRMATION]
    elif status_filter == "all":
        statuses = list(MemoryStatus)
    else:
        statuses = [MemoryStatus(status_filter)]
    memories = await _store(request).list_memories(
        user.id, statuses=statuses, now=datetime.now(UTC)
    )
    return MemoryList(memories=[MemoryOut.of(m) for m in memories])


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: uuid.UUID, request: Request, user: CurrentUser) -> MemoryOut:
    memory = await _store(request).get(user.id, memory_id)
    if memory is None:
        raise memory_not_found()
    return MemoryOut.of(memory)


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: uuid.UUID, request: Request, user: CurrentUser) -> Response:
    if not await _store(request).delete(user.id, memory_id):
        raise memory_not_found()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/memories")
async def delete_all_memories(request: Request, user: CurrentUser) -> DeletedCount:
    return DeletedCount(deleted=await _store(request).delete_all(user.id))


@router.post("/memories/{memory_id}/confirm")
async def confirm_memory(memory_id: uuid.UUID, request: Request, user: CurrentUser) -> MemoryOut:
    memory = await _store(request).confirm(user.id, memory_id)
    if memory is None:
        raise memory_not_found()
    return MemoryOut.of(memory)


@router.get("/memory-settings")
async def get_memory_settings(request: Request, user: CurrentUser) -> MemorySettingsBody:
    settings = await _store(request).get_settings(user.id)
    return MemorySettingsBody(
        memory_enabled=settings.memory_enabled, allow_sensitive=settings.allow_sensitive
    )


@router.put("/memory-settings")
async def put_memory_settings(
    body: MemorySettingsBody, request: Request, user: CurrentUser
) -> MemorySettingsBody:
    await _store(request).put_settings(
        user.id,
        MemorySettings(memory_enabled=body.memory_enabled, allow_sensitive=body.allow_sensitive),
    )
    return body
