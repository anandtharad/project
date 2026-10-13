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
  `temperature` and `reasoning_effort`, plus `response_format` on structured
  requests.
- **Ollama / openai_compatible:** the base URL is operator configuration.
  Never let end users set it (SSRF).

## Structured output (JSON Schema replies)

A request with a `response_schema` (`ResponseSchema(name, json_schema,
strict)`) asks for a JSON reply. Each adapter uses the strongest mode the
provider and model officially support, and reports it in
`GenerationResponse.structured_mode`:

- `native_strict`: the provider enforces the schema (constrained decoding).
  Only used when `strict` is true.
- `native`: the provider's best-effort schema mode, or its JSON-object mode
  (then the schema is also appended to the system instruction).
- `prompted`: no native support; a fixed block appended to the system
  instruction asks for only a JSON object matching the schema (the schema
  JSON is included).

`content` is the raw JSON text after reasoning removal; a Markdown code fence
around the whole reply is removed. Adapters never parse or validate the JSON:
the caller validates it (every mode can still return semantically wrong
values, and a refusal or a `max_tokens` cut-off can return non-JSON text).
Requests without a `response_schema` are sent exactly as before.

`strict: false` never asks for constrained decoding: providers with an
OpenAI-style `strict` flag get `strict: false` (`native`), Gemini gets JSON
mode with the schema in the instruction (`native`), and Anthropic, whose only
native mode is constrained decoding, is prompted. The table shows
`strict: true`.

| Provider | Request | Mode | Official documentation |
|---|---|---|---|
| `anthropic` | `output_config.format` (`json_schema`), GA, no beta header, on the listed models (Claude Opus 5.5 / 5 / 4.8-4.5, Sonnet 5.5 / 5 / 4.6 / 4.5, Haiku 5.5 / 4.5, Fable, Mythos); other models: instruction | `native_strict`; other models `prompted` | https://platform.claude.com/docs/en/build-with-claude/structured-outputs |
| `openai` | `response_format` `json_schema`, `strict: true`; `gpt-3.5-turbo*`, `gpt-4-*`, `gpt-4o-2024-05-13`: `json_object` + instruction | `native_strict`; legacy models `native` | https://developers.openai.com/api/docs/guides/structured-outputs |
| `groq` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`: `json_schema` `strict: true`; `openai/gpt-oss-safeguard-20b`: `json_schema` `strict: false`; every other model: `json_object` + instruction | `native_strict`; safeguard and others `native` | https://console.groq.com/docs/structured-outputs |
| `gemini` | Gemini 2.5 / 3.x: `responseMimeType: application/json` + `responseJsonSchema`; other models (2.0 needs `propertyOrdering`, Gemma is not listed): instruction | `native_strict`; other models `prompted` | https://ai.google.dev/gemini-api/docs/generate-content/structured-output |
| `xai` | `response_format` `json_schema`, `strict: true` ("guaranteed to match your schema" for supported features) | `native_strict` | https://docs.x.ai/developers/model-capabilities/text/structured-outputs |
| `qwen` | `response_format` `json_object` + instruction (JSON Schema mode exists only for qwen3.7/3.8 Plus/Flash/Max and is not available in the Singapore region the preset uses) | `native` | https://www.alibabacloud.com/help/en/model-studio/json-mode |
| `deepseek` | `response_format` `json_object` + instruction (`json_schema` is not an accepted value) | `native` | https://api-docs.deepseek.com/guides/json_mode |
| `openrouter` | `response_format` `json_schema` (strict as requested) + instruction + `provider.require_parameters: true` | `native` (enforcement varies by upstream endpoint) | https://openrouter.ai/docs/features/structured-outputs, https://openrouter.ai/docs/guides/routing/provider-selection |
| `together` | `response_format` `json_schema` + instruction (the docs ask for both) | `native` (supported models only) | https://docs.together.ai/docs/inference/chat/structured-outputs |
| `mistral` | `response_format` `json_schema`, `strict: true`; `codestral-mamba`: instruction | `native_strict`; codestral-mamba `prompted` | https://docs.mistral.ai/studio/conversations/structured-output/custom, https://docs.mistral.ai/api |
| `ollama` | `response_format` `json_schema` + instruction (the docs advise repeating the schema in the prompt; Ollama's cloud does not support structured outputs) | `native` | https://docs.ollama.com/capabilities/structured-outputs, https://docs.ollama.com/api/openai-compatibility |
| `openai_compatible` | instruction only (unknown server) | `prompted` | none (safe fallback) |
| `fake` | a deterministic minimal instance of the schema (required properties; `""`, `[]`, `0`, `false`; first enum value; `null` when nullable) | `native` (stub) | none |

The openai_compatible choices are the preset field `structured_output` in
`app/llm/providers/registry.py` (per-model exceptions in
`_MODEL_STRUCTURED_OUTPUT`); the native adapters decide per model in their
own module (`structured_plan`).

Provider notes:

- **Groq:** strict mode needs every property listed in `required` and
  `additionalProperties: false` on every object. Its documented subset:
  string, number, integer, boolean, object, array, `enum`, `anyOf` (each
  branch must follow the same rules), type arrays with `"null"` for optional
  values (`{"type": ["string", "null"]}`), and `$defs` / `$ref` (recursion
  allowed). The docs do not mention `minItems` / `maxItems`, `minLength` /
  `maxLength`, `pattern`, `format`, `minimum` / `maximum`, `const`, `oneOf`
  or `allOf`, so schemas must not rely on them in strict mode (validate those
  constraints in the caller). The state-extraction schema (a type array with
  `"null"`, `anyOf` with a `null` branch, a string `enum`, an array of
  strings) uses only documented features; both nullable forms are
  documented, so no rewrite is needed. Streaming and tool use are not
  supported together with structured outputs (this adapter uses neither).
  The reasoning controls are unchanged: gpt-oss keeps
  `include_reasoning: false`, qwen keeps `reasoning_format: "hidden"` (JSON
  modes require `parsed` or `hidden`).
- **Anthropic:** at most 16 properties with union types (`anyOf` or type
  arrays) and 24 optional properties per request; every object must set
  `additionalProperties: false`. Server-side refusal fallback stays enabled:
  on a non-streaming request the fallback model answers from scratch (the
  declined partial output is omitted), so the reply is one model's JSON. The
  docs require a request to be valid for every fallback model; the default
  routing targets are not listed, but every current model supports JSON
  outputs. If that ever fails, it surfaces as `invalid_request` and
  `GENERATION_REFUSAL_FALLBACK=off` removes the fallback.
- **Gemini:** the REST reference now prefers `generationConfig.responseFormat`
  and marks `responseJsonSchema` deprecated; the installed google-genai
  2.29.0 has no `response_format` on `GenerateContentConfig`, so the
  documented `response_mime_type` + `response_json_schema` pair is used.
- **Qwen:** the docs advise not setting `max_tokens` with structured output;
  the preset keeps its output cap, and a cut-off reply is reported as
  `max_tokens`. Models labelled "non-thinking mode" (including `qwen-plus`)
  accept `json_object` while thinking but may return invalid JSON, so leave
  thinking off for extraction.
- **OpenRouter:** without `require_parameters`, a model whose endpoints lack
  structured outputs would get the request with `response_format` silently
  ignored. With it, such a model fails with a provider error instead; pick a
  model listed with `structured_outputs` support.
- **Errors:** a schema the provider rejects is an HTTP 400 and is classified
  as `invalid_request` (not retried), as is Groq's best-effort failure
  (`Generated JSON does not match the expected schema`).

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

**Live status (2026-10-13): Groq only.** Strict structured output for state
extraction worked live on `openai/gpt-oss-120b` (5/5 valid) and
`qwen/qwen3.8-27b` (3/3 when not rate-limited). **`openai/gpt-oss-20b` failed
5/5**: although Groq lists it for strict mode, the model omitted a required
property and Groq rejected its own output with HTTP 400
`json_validate_failed`, so strict mode is not constrained decoding there in
practice. The app therefore extracts state with the reply model by default.
Every other provider is tested offline only: request shapes through each
real SDK, and the table above was checked against the official docs fetched
directly on 2026-10-12.
Facts that docs could not settle, and the choice made:

- Ollama's enforcement through the OpenAI-compatible `response_format` is
  not described (only "structured outputs work through the
  OpenAI-compatible API"), so it is treated as best effort (`native`).
- Together lists structured-output support per model in its catalogs; how an
  unsupported model reacts is not documented, so the schema also goes in the
  instruction (`native`).
- OpenRouter does not document the status code returned when no endpoint
  satisfies `require_parameters`; it surfaces with the usual classification.
- Mistral's API reference says JSON schema mode "guarantees" the schema but
  does not describe the mechanism; it is reported as `native_strict` on that
  wording.
- Anthropic does not list the default server-side fallback targets (see the
  note above).
