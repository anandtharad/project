"""Model-specific exports of canonical training records (spec §17.3 "model-
specific chat template", Phase 7 task 7). Formats were checked against the
official documentation on 2026-10-16 (docs/fine-tuning.md lists the sources):

- `trl`: Hugging Face TRL conversational prompt-completion rows,
  {"prompt": [messages], "completion": [assistant message]}. The trainer
  applies the model's own chat template, and by default trains on the
  completion only. This is the format to prefer: the template is the
  model's, not a copy.
- `openai`: OpenAI chat fine-tuning lines, {"messages": [...]}; every
  assistant message carries "weight" (1 for the target, 0 for the history),
  since no documented default exists.
- `chatml`: text rendered with the Qwen2.5 ChatML template.
- `llama3`: text rendered with the Meta Llama 3 template. Not for Llama 3.1,
  whose official template adds date lines to the system message: use `trl`.

Rendered formats are {"prompt": text ending with the assistant header,
"completion": target and the end-of-turn token}.

The system message is the companion prompt the record references, with the
policy fragments a turn would get: the record's strategy when it has one,
otherwise the policy chosen for the message (app/orchestration/policy.py),
via the engine's own `system_prompt()`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.decisions.schemas import DecisionSet
from app.orchestration.engine import system_prompt
from app.orchestration.policy import ResponsePolicy, Strategy, choose_policy, explicit_request
from app.prompting import PromptRegistry, PromptTemplate
from app.training.schema import TrainingRecord

NO_DECISIONS = DecisionSet(provider="off", results={})
FORMATS = ("trl", "openai", "chatml", "llama3")


def policy_for(record: TrainingRecord) -> ResponsePolicy:
    if record.strategy is None:
        return choose_policy(record.message, NO_DECISIONS)
    _, advice_allowed = explicit_request(record.message)
    return ResponsePolicy(
        strategy=Strategy(record.strategy.value), advice_allowed=advice_allowed, source="decision"
    )


def system_text(record: TrainingRecord, registry: PromptRegistry) -> str:
    prompt_id, _, version = record.system.prompt.partition("@")
    prompt: PromptTemplate = registry.get(prompt_id, version)
    if prompt.checksum != record.system.checksum:
        raise ValueError(f"{record.system.prompt} changed since the build (checksum differs)")
    return system_prompt(prompt, policy_for(record))


def prompt_messages(record: TrainingRecord, system: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        *({"role": m.role, "content": m.content} for m in record.history),
        {"role": "user", "content": record.message},
    ]


def trl_row(record: TrainingRecord, system: str) -> dict[str, Any]:
    return {
        "prompt": prompt_messages(record, system),
        "completion": [{"role": "assistant", "content": record.target}],
    }


def openai_row(record: TrainingRecord, system: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in prompt_messages(record, system):
        if message["role"] == "assistant":
            messages.append({**message, "weight": 0})
        else:
            messages.append(message)
    messages.append({"role": "assistant", "content": record.target, "weight": 1})
    return {"messages": messages}


def chatml_row(record: TrainingRecord, system: str) -> dict[str, Any]:
    # Qwen2.5: <|im_start|>{role}\n{content}<|im_end|>\n per turn; with a
    # system message first, no default system text is inserted; the
    # generation prompt is <|im_start|>assistant\n; EOS is <|im_end|>.
    turns = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        for m in prompt_messages(record, system)
    )
    return {"prompt": turns + "<|im_start|>assistant\n", "completion": record.target + "<|im_end|>"}


def llama3_row(record: TrainingRecord, system: str) -> dict[str, Any]:
    # Meta Llama 3: <|begin_of_text|> once, then per turn
    # <|start_header_id|>{role}<|end_header_id|>\n\n{content stripped}<|eot_id|>,
    # ending with the open assistant header; the reply ends with <|eot_id|>.
    turns = "".join(
        f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>"
        for m in prompt_messages(record, system)
    )
    return {
        "prompt": "<|begin_of_text|>" + turns + "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "completion": record.target.strip() + "<|eot_id|>",
    }


ROWS: dict[str, Callable[[TrainingRecord, str], dict[str, Any]]] = {
    "trl": trl_row,
    "openai": openai_row,
    "chatml": chatml_row,
    "llama3": llama3_row,
}


def export(
    records: list[TrainingRecord], fmt: str, registry: PromptRegistry | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Rows per split, in record order."""
    registry = registry or PromptRegistry()
    row = ROWS[fmt]
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for record in records:
        out[record.split].append(row(record, system_text(record, registry)))
    return out
