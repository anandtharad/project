"""Voice over WebSocket (docs/voice-interface.md).

    POST /v1/conversations/{id}/voice/tickets   -> {"ticket", "expires_in", "path"}
    WS   /v1/voice?ticket=...                    -> a VoiceSession

Client -> server, after connecting:
- binary frames: the person's audio, 16-bit mono PCM at VOICE_INPUT_SAMPLE_RATE;
- {"type": "start", "output": ["text", "audio"]}  (optional, first; choose outputs);
- {"type": "end_of_speech"}                        (push-to-talk released);
- {"type": "text", "content": "..."}               (a typed message);
- {"type": "playback_done"}                        (the reply's audio finished playing);
- {"type": "stop"}.
Server -> client: JSON events ("ready", "state", "transcript", "reply",
"audio_end", "no_speech", "interrupted", "error") and binary frames with the
reply's audio (PCM, format in "ready").

Close codes: 4401 invalid ticket, 4503 voice disabled, 4429 too many open
sessions for this person, 4408 no message from the client for
VOICE_IDLE_TIMEOUT_SECONDS.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from app.api.auth import AuthenticatedUser, authenticate
from app.api.messages import safety_notice
from app.api.schemas import CitedSource, SubmitMessageRequest
from app.config import Settings
from app.db import repositories as repo
from app.errors import AppError, ErrorCategory, conversation_not_found
from app.observability import log_event
from app.orchestration.engine import TurnOrchestrator
from app.voice.base import AudioFormat
from app.voice.limits import VoiceLimits
from app.voice.session import TurnReply, VoiceConfig, VoiceSession

logger = logging.getLogger("app.voice")
router = APIRouter(tags=["voice"])
CurrentUser = Annotated[AuthenticatedUser, Depends(authenticate)]

CLIENT_PAGE = Path(__file__).resolve().parent.parent / "static" / "voice.html"
MAX_AUDIO_FRAME_BYTES = 64 * 1024
MAX_TEXT_FRAME_BYTES = 64 * 1024
START_WAIT_SECONDS = 2.0
# WebSocket close codes in the application range.
CLOSE_UNAUTHORIZED = 4401
CLOSE_UNAVAILABLE = 4503
CLOSE_TOO_MANY_SESSIONS = 4429
CLOSE_IDLE = 4408


class VoiceTicket(BaseModel):
    ticket: str
    expires_in: int
    path: str


def _too_many_sessions() -> AppError:
    return AppError(
        ErrorCategory.RATE_LIMITED,
        "too_many_voice_sessions",
        "Too many voice sessions are open; close one first.",
        http_status=429,
    )


def _voice_disabled() -> AppError:
    return AppError(
        ErrorCategory.INVALID_REQUEST,
        "voice_disabled",
        "Voice is not enabled on this server (VOICE_ENABLED).",
        http_status=409,
    )


@router.get("/voice", include_in_schema=False)
async def voice_client(request: Request) -> FileResponse:
    """A development page for talking to the companion (text and voice)."""
    if not request.app.state.settings.voice_enabled:
        raise _voice_disabled()
    return FileResponse(CLIENT_PAGE, media_type="text/html", headers={"Cache-Control": "no-store"})


@router.post(
    "/v1/conversations/{conversation_id}/voice/tickets", status_code=status.HTTP_201_CREATED
)
async def create_voice_ticket(
    conversation_id: uuid.UUID, request: Request, user: CurrentUser
) -> VoiceTicket:
    settings: Settings = request.app.state.settings
    if not settings.voice_enabled:
        raise _voice_disabled()
    async with request.app.state.session_factory() as session:
        if await repo.get_owned_conversation(session, conversation_id, user.id) is None:
            raise conversation_not_found()
    if not request.app.state.voice_limits.can_open(user.id):
        raise _too_many_sessions()
    ticket = request.app.state.voice_tickets.issue(user.id, conversation_id)
    return VoiceTicket(
        ticket=ticket, expires_in=settings.voice_ticket_ttl_seconds, path="/v1/voice"
    )


@router.websocket("/v1/voice")
async def voice_socket(websocket: WebSocket, ticket: str = "") -> None:
    app = websocket.app
    settings: Settings = app.state.settings
    # Accept, then close with a code: a refused handshake is a bare HTTP 403
    # that browsers do not expose to the page.
    await websocket.accept()
    if not settings.voice_enabled:
        await websocket.close(code=CLOSE_UNAVAILABLE, reason="voice_disabled")
        return
    redeemed = app.state.voice_tickets.redeem(ticket) if ticket else None
    if redeemed is None:
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="invalid_ticket")
        return
    limits: VoiceLimits = app.state.voice_limits
    if not limits.can_open(redeemed.user_id):
        await websocket.close(code=CLOSE_TOO_MANY_SESSIONS, reason="too_many_sessions")
        return
    limits.opened(redeemed.user_id)
    try:
        await _serve(websocket, settings, redeemed.user_id, redeemed.conversation_id)
    finally:
        limits.closed(redeemed.user_id)


async def _serve(
    websocket: WebSocket, settings: Settings, user_id: uuid.UUID, conversation_id: uuid.UUID
) -> None:
    app = websocket.app
    limits: VoiceLimits = app.state.voice_limits
    orchestrator: TurnOrchestrator = app.state.orchestrator
    stt, tts = app.state.voice_stt, app.state.voice_tts

    async def run_turn(text: str, modality: str) -> TurnReply:
        # The same rules as the HTTP API's message body (no NUL, not empty).
        try:
            SubmitMessageRequest(content=text)
        except ValidationError:
            raise AppError(
                ErrorCategory.INVALID_REQUEST,
                "validation_failed",
                "The message is empty or contains characters that are not allowed.",
                http_status=422,
            ) from None
        if len(text) > settings.max_input_characters:
            raise AppError(
                ErrorCategory.INVALID_REQUEST,
                "content_too_long",
                f"content must be at most {settings.max_input_characters} characters.",
                http_status=422,
            )
        metadata: dict[str, Any] = {"modality": modality, "channel": "voice_socket"}
        if modality == "voice" and stt is not None:
            metadata["stt"] = {"provider": stt.name, "model": stt.model}
        result = await orchestrator.submit_message(
            trace_id=uuid.uuid4().hex,
            user_id=user_id,
            conversation_id=conversation_id,
            content=text,
            idempotency_key=f"voice-{uuid.uuid4().hex}",
            input_metadata=metadata,
        )
        turn = result.turn
        notice = safety_notice(turn)
        return TurnReply(
            text=turn.assistant_content,
            user_message_id=str(turn.user_message_id),
            assistant_message_id=str(turn.assistant_message_id),
            extra={
                "safety": notice.model_dump() if notice else None,
                "sources": [
                    CitedSource.model_validate(s).model_dump(mode="json") for s in turn.sources
                ],
            },
        )

    send_lock = asyncio.Lock()

    async def send(event: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_text(json.dumps(event, default=str))

    async def send_audio(pcm: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(pcm)

    # The client may choose its outputs in a first "start" message.
    outputs = {"text", "audio"}
    first: MutableMapping[str, Any] | None = None
    with contextlib.suppress(TimeoutError):
        first = await asyncio.wait_for(websocket.receive(), START_WAIT_SECONDS)
    if first is not None and first.get("type") == "websocket.disconnect":
        return
    first_text = first.get("text") if first is not None else None
    # Size first: a huge frame must not be parsed on the event loop.
    start = _json(first_text) if _small(first_text) else None
    if start is not None and start.get("type") == "start" and isinstance(start.get("output"), list):
        outputs = {str(o) for o in start["output"]} & {"text", "audio"} or {"text"}
        first = None

    session = VoiceSession(
        config=VoiceConfig(
            language=settings.voice_language,
            input_audio=AudioFormat(sample_rate=settings.voice_input_sample_rate),
            duplex=settings.voice_duplex,
            end_of_turn=settings.voice_end_of_turn,
            voice=settings.tts_voice,
            max_utterance_seconds=settings.voice_max_utterance_seconds,
        ),
        stt=stt,
        tts=tts,
        run_turn=run_turn,
        send=send,
        send_audio=send_audio,
        audio_output="audio" in outputs,
        charge_audio=lambda seconds: limits.charge_audio(user_id, seconds),
    )
    log_event(
        logger,
        "voice.connected",
        conversation_id=str(conversation_id),
        stt=stt.name if stt else None,
        tts=tts.name if tts else None,
        outputs=sorted(outputs),
    )
    try:
        await session.start()
        if first is not None and await _dispatch(session, first) is False:
            return
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), settings.voice_idle_timeout_seconds
                )
            except TimeoutError:
                await websocket.close(code=CLOSE_IDLE, reason="idle")
                return
            if message.get("type") == "websocket.disconnect":
                break
            if await _dispatch(session, message) is False:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        # The session handles its own failures; this is a backstop that keeps
        # the reason in the log instead of a bare ASGI traceback.
        logger.exception("voice.session_failed")
    finally:
        await session.close()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _dispatch(session: VoiceSession, message: MutableMapping[str, Any]) -> bool:
    """Route one received frame; False to end the session."""
    data = message.get("bytes")
    if data is not None:
        if len(data) > MAX_AUDIO_FRAME_BYTES or len(data) % 2:
            await session.notify(
                {
                    "type": "error",
                    "code": "bad_audio_frame",
                    "message": "Send 16-bit PCM frames under 64 KB.",
                }
            )
            return True
        await session.on_audio(data)
        return True
    text = message.get("text")
    if not _small(text):
        return True
    event = _json(text)
    kind = event.get("type") if event else None
    if kind == "end_of_speech":
        await session.on_end_of_speech()
    elif kind == "text" and event is not None and isinstance(event.get("content"), str):
        await session.on_text(event["content"])
    elif kind == "playback_done":
        await session.on_playback_done()
    elif kind == "stop":
        return False
    else:
        await session.notify(
            {"type": "error", "code": "bad_message", "message": "Unknown message."}
        )
    return True


def _small(text: Any) -> bool:
    return isinstance(text, str) and len(text) <= MAX_TEXT_FRAME_BYTES


def _json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
