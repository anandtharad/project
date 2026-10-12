# LLM providers

The generative LLM is selected with `GENERATION_PROVIDER`. Each provider is a
**preset** (`app/llm/providers/registry.py`) that names an adapter, endpoint,
API-key variable, default model, and request shape. `GENERATION_MODEL`,
`GENERATION_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_EFFORT`,
`MAX_OUTPUT_TOKENS` and `MAX_CONTEXT_TOKENS` override the preset.

Default models are starting points, not measured choices (spec §5.2). Every
reply records the model that actually served it (`messages.model_id`).

This build covers text only. Speech-to-text and text-to-speech (for example
Sarvam or ElevenLabs) are separate integrations and not part of these adapters.

## Supported providers

| `GENERATION_PROVIDER` | SDK (extra) | Default model | Key variable | Effort control |
|---|---|---|---|---|
| `anthropic` | `anthropic` | `claude-opus-5-5` | `ANTHROPIC_API_KEY` (or SDK profile) | `output_config.effort` (`none` → `low`) |
| `openai` | `openai` | `gpt-6-luna` | `OPENAI_API_KEY` | `reasoning_effort` (all levels) |
| `groq` | `groq` | `openai/gpt-oss-120b` | `GROQ_API_KEY` | per model family (below) |
| `gemini` | `google-genai` (`gemini`) | `gemini-3.6-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `thinking_level` (3.x) / `thinking_budget` (2.5) |
| `xai` (Grok) | `openai` | `grok-4.3` | `XAI_API_KEY` | `reasoning_effort` none/low/medium/high |
| `qwen` (Alibaba Model Studio) | `openai` | `qwen-plus` | `DASHSCOPE_API_KEY` | `enable_thinking` + `thinking_budget` |
| `deepseek` | `openai` | (set `GENERATION_MODEL`) | `DEEPSEEK_API_KEY` | `reasoning_effort` none/low/high/max |
| `openrouter` | `openai` | (set `GENERATION_MODEL`) | `OPENROUTER_API_KEY` | not mapped (warning) |
| `together` | `openai` | (set `GENERATION_MODEL`) | `TOGETHER_API_KEY` | not mapped (warning) |
| `mistral` | `openai` | `mistral-large-latest` | `MISTRAL_API_KEY` | `reasoning_effort` none/high |
| `ollama` (local) | `openai` | (set `GENERATION_MODEL`) | none | `reasoning_effort` (model-defined) |
| `openai_compatible` | `openai` | (set `GENERATION_MODEL`) | `GENERATION_API_KEY` (optional) | not mapped (warning) |
| `fake` | none | `fake-stub` | none | development stub, no model call |

Install SDKs with `uv sync --all-extras`, or only what you need
(`uv sync --extra groq`). If the selected provider's SDK is missing, startup
fails with the exact install command.

`GENERATION_EFFORT` takes `none | low | medium | high | xhigh | max`, or `off`
to send no effort parameter at all (for models without reasoning controls).
A value a provider or model does not accept is clamped to the nearest accepted
value, and the service logs a `config.warning` at startup (for example `none`
becomes `low` on `grok-4.7`, which cannot turn reasoning off).

When `GENERATION_EFFORT` is unset, the anthropic, openai, groq and gemini
presets use `low`, but only for models whose controls are known: the
preset's default model, Groq's gpt-oss and qwen families, and Gemini 2.5/3.x.
Other models get no effort parameter unless you set one.

Startup checks (fail fast with a clear message):
- `GENERATION_TEMPERATURE` above 1 with Anthropic (it accepts 0-1).
- `GENERATION_TEMPERATURE` with OpenAI reasoning effort other than `none`
  (reasoning models reject it); use `GENERATION_EFFORT=none`, or `off` for
  non-reasoning models.
- `GENERATION_API_KEY` and the provider's own key both set to different
  values, so a stale key is never sent to the wrong provider.

## Groq (free tier)

Current free-tier chat models, with Groq's published free-tier limits (about
30 requests/min, 1K requests/day, 8K tokens/min and 200K tokens/day per model):

- `openai/gpt-oss-120b`: default; production; strongest.
- `openai/gpt-oss-20b`: production; fastest.
- `qwen/qwen3.8-27b`: preview only; may be withdrawn at short notice.

Groq retired its Llama, Qwen3-32B and Kimi chat models for the free and
developer tiers in 2026, so they are not offered.

Request shape by model family:
- gpt-oss: `include_reasoning=false`; `reasoning_effort` low/medium/high
  (`none` maps to `low`, because gpt-oss always reasons).
- qwen: `reasoning_format="hidden"` (otherwise reasoning appears inline as
  `<think>`); `reasoning_effort` none/low/medium/high.

Free-tier defaults: `MAX_OUTPUT_TOKENS=2048` and `MAX_CONTEXT_TOKENS=4000`, so
one turn fits inside a minute's token budget. Raise them on a paid tier.

## Notes per provider

- **OpenAI:** the instruction is sent as a `developer` message, and output is
  capped with `max_completion_tokens` (covers reasoning tokens).
- **xAI:** officially OpenAI-SDK compatible at `https://api.x.ai/v1`. Retired
  Grok slugs are served by `grok-4.3`, so check `model_id`. An invalid key is
  reported as HTTP 400 and classified as an authentication error.
- **Qwen:** uses the Singapore endpoint by default. Keys are bound to a
  region, so set `GENERATION_BASE_URL` for Beijing or US workspaces.
  `qwen-plus` does not think by default, which keeps non-streaming calls
  valid; qwen3.5+ models think by default, and the open-source Qwen3 models
  only stream while thinking. Provider moderation (`data_inspection_failed`)
  is reported as `content_filtered`.
- **Gemini:** the native SDK is used, not Gemini's OpenAI-compatible
  endpoint. A per-minute 429 honors the server's retry delay; an exhausted
  daily quota is `quota_exhausted` and is not retried.
- **Mistral:** Mistral documents the OpenAI SDK path, and its API rejects
  unknown fields; this adapter sends only `model`, `messages`, `max_tokens`,
  `temperature` and `reasoning_effort`.
- **Ollama / openai_compatible:** the base URL is operator configuration.
  Never let end users set it (SSRF).

## Credential safety (OpenAI SDK)

The `openai` SDK reads several `OPENAI_*` variables implicitly. The adapter
counters each one, and tests cover all of them:

- `OPENAI_API_KEY` is used when no key is passed, even for third-party base
  URLs. The adapter always passes an explicit key (a placeholder for keyless
  servers).
- `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` / `OPENAI_CUSTOM_HEADERS` are
  forwarded to any endpoint. For non-OpenAI endpoints those headers are
  removed.
- An `Authorization` line in `OPENAI_CUSTOM_HEADERS` overrides even an
  explicit key. The adapter pins `Authorization` to the configured key.
- `OPENAI_BASE_URL` redirects the default endpoint. Every preset passes an
  explicit base URL.

## Verification status

Every adapter is tested offline through its real SDK with a mocked HTTP
transport: request JSON and headers, response normalization, reasoning
removal, error classification and retries. All request shapes were checked
against official documentation and the installed SDK source. Many official
docs pages were only reachable through search summaries from this
environment; the contract notes record which facts that affects.

**Groq is verified live** (2026-10-12): the gated live suite passed 6/6 on
the user's machine for `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
`qwen/qwen3.8-27b`. **Other providers have not had a live call** (no keys;
this cloud environment's egress policy also blocks provider APIs). The live
suite runs the same way on Windows, macOS and Linux and reads `GROQ_API_KEY`
from `.env` or the environment:

```bash
uv run python scripts/dev.py live-groq
```

Plain `pytest` skips it (opt-in). The command above sets the opt-in.
