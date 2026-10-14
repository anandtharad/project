"""System One decision backend on the official `typesafe-sdk`.

One adapter serves two providers that speak the same wire protocol
(`POST /v1/systemone`):

- jev:  TypeSafe's hosted API (https://api.typesafe.ai).
- laya: a self-hosted Laya server (`laya-serve`), which implements the Jev
        request and response shape. Run it with LAYA_JEV_STRICT=1 so the
        response matches the Jev contract exactly.

The third provider, llm, is a general LLM behind the same interface
(`app/decisions/llm_backend.py`).

The SDK reads TYPESAFE_API_KEY / TYPESAFE_BASE_URL / TYPESAFE_DEFAULT_MODEL
when an argument is missing; this adapter always passes all three, so no
ambient variable can redirect a request or attach another key. The SDK logs
request and response bodies at DEBUG, so its logger is capped at WARNING.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from app.decisions.schemas import DecisionProvider, FallbackReason, QuestionSpec

if TYPE_CHECKING:
    import httpx2

SDK_LOGGER = "typesafe_sdk"


class DecisionBackendError(Exception):
    """The request produced no usable response. `reason` and `message` are safe
    to log (never content).

    For rate limits, `retry_after_seconds` is the provider's requested wait and
    `quota_exhausted` says a daily or billing quota is used up (retrying soon
    will not help); the evaluation script uses both to pace a run.
    """

    def __init__(
        self,
        reason: FallbackReason,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.quota_exhausted = quota_exhausted


@dataclass(frozen=True)
class BackendAnswers:
    """Raw answers keyed by question id, as plain JSON-like values."""

    model_id: str
    answers: dict[str, dict[str, Any]]
    latency_ms: float
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None = None


def cap_sdk_logger() -> None:
    # The SDK applies TYPESAFE_LOG_LEVEL when it is imported, which can lower
    # the level after logging was configured; cap it again after import.
    sdk_logger = logging.getLogger(SDK_LOGGER)
    if sdk_logger.getEffectiveLevel() < logging.WARNING:
        sdk_logger.setLevel(logging.WARNING)


class SystemOneBackend:
    def __init__(
        self,
        *,
        provider: DecisionProvider,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        import typesafe_sdk

        cap_sdk_logger()
        self.provider = provider
        self.model = model
        self._sdk = typesafe_sdk
        self._client = typesafe_sdk.AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout_seconds,
            # The retry budget stays inside the per-turn decision timeout.
            retry=typesafe_sdk.RetryPolicy(
                max_retries=max_retries, backoff_max=1.0, timeout=timeout_seconds
            ),
            http_client=http_client,
        )

    async def ask(self, state: Mapping[str, Any], questions: list[QuestionSpec]) -> BackendAnswers:
        sdk = self._sdk
        started = time.perf_counter()
        try:
            response = await self._client.system_one(
                state=dict(state),
                # Raw question dictionaries are a documented SDK input (QuestionModel).
                questions={q.id: cast(Any, q.wire()) for q in questions},
            )
        except sdk.TypeSafeAPITimeoutError as exc:
            raise DecisionBackendError(
                FallbackReason.TIMEOUT, "decision request timed out"
            ) from exc
        except sdk.TypeSafeAPIConnectionError as exc:
            raise DecisionBackendError(
                FallbackReason.UNAVAILABLE, "could not reach the decision service"
            ) from exc
        except sdk.TypeSafeAPIResponseValidationError as exc:
            raise DecisionBackendError(
                FallbackReason.INVALID_RESPONSE,
                f"response did not match the protocol at {exc.field_path!r}",
                status_code=exc.status,
            ) from exc
        except sdk.TypeSafeRateLimitError as exc:
            raise DecisionBackendError(
                FallbackReason.RATE_LIMITED,
                "decision service rate limit",
                status_code=exc.status,
                retry_after_seconds=(
                    exc.retry_after_ms / 1000 if exc.retry_after_ms is not None else None
                ),
            ) from exc
        except sdk.TypeSafeAPIError as exc:
            # 529 (overloaded) and other 5xx are outages; 4xx are rejections.
            reason = FallbackReason.UNAVAILABLE if exc.status >= 500 else FallbackReason.REJECTED
            raise DecisionBackendError(
                reason, f"decision service returned HTTP {exc.status}", status_code=exc.status
            ) from exc
        except sdk.TypeSafeError as exc:
            # Raised before sending (e.g. an invalid question); a bug, not an outage.
            raise DecisionBackendError(FallbackReason.REJECTED, type(exc).__name__) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        answers = {
            name: answer.model_dump(mode="json") for name, answer in response.answers.items()
        }
        request_id = response.raw_http_response.headers.get("x-typesafe-request-id")
        return BackendAnswers(
            model_id=response.model,
            answers=answers,
            latency_ms=latency_ms,
            request_id=request_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
