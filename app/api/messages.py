from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status

from app.api.auth import AuthenticatedUser, authenticate
from app.api.schemas import (
    MessageList,
    MessageOut,
    SafetyNotice,
    SubmitMessageRequest,
    SubmitMessageResponse,
    SupportContact,
)
from app.config import Settings
from app.db import repositories as repo
from app.db.models import MessageRole
from app.errors import AppError, ErrorCategory, conversation_not_found
from app.observability import trace_id_var
from app.orchestration.engine import TurnOrchestrator
from app.orchestration.persistence import CompletedTurn
from app.safety.policy import PATHWAY_RULES
from app.safety.resources import default_resources
from app.safety.taxonomy import Pathway

router = APIRouter(prefix="/v1/conversations/{conversation_id}/messages", tags=["messages"])

CurrentUser = Annotated[AuthenticatedUser, Depends(authenticate)]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Replay of a completed request (same idempotency key)."}},
)
async def submit_message(
    conversation_id: uuid.UUID,
    body: SubmitMessageRequest,
    request: Request,
    response: Response,
    user: CurrentUser,
) -> SubmitMessageResponse:
    settings: Settings = request.app.state.settings
    if len(body.content) > settings.max_input_characters:
        raise AppError(
            ErrorCategory.INVALID_REQUEST,
            "content_too_long",
            f"content must be at most {settings.max_input_characters} characters.",
            http_status=422,
        )
    orchestrator: TurnOrchestrator = request.app.state.orchestrator
    result = await orchestrator.submit_message(
        trace_id=trace_id_var.get() or uuid.uuid4().hex,
        user_id=user.id,
        conversation_id=conversation_id,
        content=body.content,
        idempotency_key=body.idempotency_key,
    )
    turn = result.turn
    if turn.replayed:
        response.status_code = status.HTTP_200_OK
    return SubmitMessageResponse(
        conversation_id=turn.conversation_id,
        user_message_id=turn.user_message_id,
        assistant_message_id=turn.assistant_message_id,
        assistant_content=turn.assistant_content,
        created_at=turn.created_at,
        replayed=turn.replayed,
        timing_ms=result.timings_ms if settings.expose_timing_metadata else None,
        safety=_safety_notice(turn),
    )


def _safety_notice(turn: CompletedTurn) -> SafetyNotice | None:
    if turn.safety_pathway is None or turn.safety_pathway == Pathway.NORMAL:
        return None
    pathway = Pathway(turn.safety_pathway)
    chosen = default_resources().select(turn.safety_region, PATHWAY_RULES[pathway].resource_kinds)
    return SafetyNotice(
        pathway=pathway.value,
        resources=[SupportContact(name=r.name, contact=r.contact, hours=r.hours) for r in chosen],
    )


@router.get("")
async def list_messages(
    conversation_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> MessageList:
    async with request.app.state.session_factory() as session:
        conversation = await repo.get_owned_conversation(session, conversation_id, user.id)
        if conversation is None:
            raise conversation_not_found()
        rows = await repo.page_messages(
            session, conversation.id, after_sequence=after_sequence, limit=limit + 1
        )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return MessageList(
        conversation_id=conversation.id,
        messages=[
            MessageOut(
                message_id=row.id,
                sequence=row.sequence,
                role="user" if row.role == MessageRole.USER else "assistant",
                content=row.content,
                created_at=row.created_at,
                status=row.turn_status,
            )
            for row in rows
        ],
        next_after_sequence=rows[-1].sequence if has_more and rows else None,
    )
