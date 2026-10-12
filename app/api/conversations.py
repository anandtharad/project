from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.api.auth import AuthenticatedUser, authenticate
from app.api.schemas import ConversationCreated, ConversationDetail
from app.db import repositories as repo
from app.errors import conversation_not_found

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

CurrentUser = Annotated[AuthenticatedUser, Depends(authenticate)]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(request: Request, user: CurrentUser) -> ConversationCreated:
    async with request.app.state.session_factory() as session, session.begin():
        conversation = await repo.create_conversation(session, user.id)
    return ConversationCreated(
        conversation_id=conversation.id,
        created_at=conversation.created_at,
        status=conversation.status,
    )


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID, request: Request, user: CurrentUser
) -> ConversationDetail:
    async with request.app.state.session_factory() as session:
        conversation = await repo.get_owned_conversation(session, conversation_id, user.id)
        if conversation is None:
            raise conversation_not_found()
        message_count = await repo.count_messages(session, conversation.id)
    return ConversationDetail(
        conversation_id=conversation.id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        status=conversation.status,
        message_count=message_count,
    )
