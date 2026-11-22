# Agent turns (model-directed strategy and tools)

`AGENT_MODE=on` lets the reply model choose how to respond and what to
look up before it replies. It is opt-in and off by default. With it off,
every turn is exactly as before.

When it is on, the agent runs only on **normal safety pathway** turns.
Crisis, check-in and other safety pathways, and turns where the policy
carries a safety signal, keep the deterministic path. The agent never
decides safety.

## What the model decides, and what the application decides

| The model decides | The application decides (code) |
|---|---|
| Which of 5 response strategies fits the latest message | Which strategies exist and their exact guidance text (locked prompt `agent_strategies`) |
| Whether to call a tool, which one, with what search words | Who the person and conversation are; what each tool can read; time and size limits |
| When it has enough to reply | How many iterations and tool calls a turn may use; the turn deadline |
| The reply text | Safety pathway, output checks, regeneration, fallbacks, citation validity, persistence |

## Flow

```text
TurnOrchestrator._run (unchanged up to the policy)
  begin (tx 1) -> state | memory pre-fetch | safety | knowledge pre-fetch -> safety decide -> policy
  safety pathway or policy.safety_signal?  -> deterministic path (as before)
  otherwise, AGENT_MODE=on:
    LangGraph (app/agent/graph.py), no checkpointer, services in the run context:
      prepare   strategy selection: one structured call, schema = enum of strategy ids;
                invalid, unknown or failed -> "conversational" (recorded as a fallback)
                system prompt = companion prompt with {strategy guidance [+ no_advice] + tool guidance}
                context = budgeted history only (no notes, memories or evidence up front)
      agent     one model call with the 4 tool definitions, tool_choice "auto"
                tool calls -> tools ; text -> finalize
      tools     each call: validate args -> run with timeout -> bound the size -> TOOL message
      agent     ... at most AGENT_MAX_ITERATIONS calls that may use tools and
                AGENT_MAX_TOOL_CALLS tool calls; then one last call with
                tool_choice "none" and the final-answer note
      finalize  reply, strategy, record -> back to the orchestrator
    any failure (provider error, deadline, tools after the limit, malformed reply,
    unexpected exception) -> deterministic path, with the pre-fetched memories and evidence
  ResponseValidator -> output safety checks (one regeneration from the agent's own
  messages, tool_choice "none") -> citations checked against the agent's evidence
  -> TurnStore.complete (tx 2) -> background memory extraction (unchanged)
```

PostgreSQL stays the only record of the conversation. Nothing the graph
holds outlives the turn. Idempotency, ordering, the one-turn-in-flight
rule and the two transactions are the turn store's, as before.

## Strategies

These are versioned fragments of `prompts/agent_strategies/1.0.0.toml`,
which is checksum-locked like every prompt.

| Id | When | Response constraints (summary) |
|---|---|---|
| `empathetic` | painful or heavy news | acknowledge with their specifics; no advice or silver linings unless asked; 1-3 sentences, at most one gentle question |
| `supportive` | wants encouragement or next steps | one sentence of acknowledgement, then 1-2 concrete options, not instructions |
| `reflective` | making sense of their own thoughts | reflect, name a pattern only if clearly there, one open question; no labels, no advice |
| `informational` | asks for facts | brief, clear; search the knowledge base and cite `[S1]`; never numbers, doses, contacts or brands from memory |
| `conversational` | ordinary chat (the default) | natural, matching their length; no unprompted feelings talk |

The selection call sees the last 4 messages (600 characters each) and the
conversation notes, escaped inside `<conversation_data>`. It answers
`{"strategy", "reason"}`. The reason is never stored, because it can quote
the person. A person's explicit "no advice" request still adds the
companion `no_advice` fragment, whatever the strategy.

## Tools

The tools are all read-only. Identity is never an argument: the person
and conversation come from the authenticated turn. The argument schemas
forbid unknown fields, so a `user_id` the model invents is rejected as
invalid.

| Tool | Arguments | Reads | Result (escaped, untrusted block) |
|---|---|---|---|
| `search_user_memories` | `query` (2-300 chars) | `MemoryService.retrieve_for_turn`: the person's active memories, with the memory settings, statuses and retention | `<memories>` |
| `search_knowledge_base` | `query` (2-300 chars) | `KnowledgeService.search`: the reviewed corpus, hybrid retrieval and relevance thresholds | `<evidence>` with markers S1.. (numbered across searches, max 8 passages) |
| `search_conversation_history` | `query` (2-300 chars) | Full-text search (`websearch_to_tsquery`, GIN index, migration 0010) over the person's own messages in all their conversations, excluding what is already in the context; completed turns only | `<conversation_history>`: up to 5 snippets of at most 300 chars, each marked `this_conversation` true or false |
| `get_conversation_state` | none | the turn's extracted conversation state | `<conversation_notes>` |

Every call:
- has a timeout (`AGENT_TOOL_TIMEOUT_SECONDS`, and never past the turn
  deadline) and a size cap (`AGENT_TOOL_MAX_RESULT_CHARACTERS`);
- is truncated inside its block, so the block stays closed;
- comes back to the model as a normalized error on failure
  (`invalid_arguments`, `unknown_tool`, `timeout`, `unavailable`, `disabled`,
  `limit_reached`), never as an invented result;
- is logged as `agent.tool` with the tool, status, duration and count,
  never the content.

History search covers all of the person's own chats (the chosen scope).
Deleting a conversation removes its messages, so they can no longer be
found.

## Configuration

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `AGENT_MODE` | `off` | `off`, `on` | agent turns on normal-pathway turns |
| `AGENT_STRATEGY_PROMPT_VERSION` | `1.0.0` | | `agent_strategies` prompt version |
| `AGENT_MAX_ITERATIONS` | 4 | 1-8 | model calls that may request tools |
| `AGENT_MAX_TOOL_CALLS` | 6 | 1-12 | tool calls per turn (calls past it get `limit_reached`) |
| `AGENT_TOOL_TIMEOUT_SECONDS` | 5.0 | | per tool call, capped by the turn deadline |
| `AGENT_TOOL_MAX_RESULT_CHARACTERS` | 4000 | 500-20000 | per tool result |

The turn deadline (`TURN_DEADLINE_SECONDS`) bounds everything. The graph
enforces it with `asyncio.timeout` around every model call, whatever the
provider does with `timeout_seconds`.

Startup refuses `AGENT_MODE=on`:
- when the reply provider has no tool calling;
- when the companion prompt has no `response_guidance` slot;
- in production, when LangSmith tracing variables are set. Elsewhere
  this only logs a warning. LangChain's tracing client comes with
  `langchain-core` and would send conversation content if switched on.

## Provider compatibility

| Provider | Tool calling | `tool_choice` | Notes |
|---|---|---|---|
| Groq (`groq`) | yes | `auto` / `none` sent | Groq docs: gpt-oss-120b/20b support tool use but not parallel calls; the graph runs calls one at a time anyway. Structured outputs and tools cannot be combined in one call, which is why strategy selection is a separate call |
| Ollama (`ollama`) | yes (`tools`) | not sent | Ollama's OpenAI layer lists `tools` as supported and `tool_choice` as not. A "no more tools" call omits the tool definitions instead. If a model still calls a tool after the limit, the turn falls back to the deterministic reply. Quality depends heavily on the local model: use one documented for tool calling (Ollama's examples use `qwen3`). Needs `OLLAMA_CONTEXT_LENGTH=8192` or more (docs/providers.md) |
| Development stub (`fake`) | accepts tools, never calls one | | for deterministic runs |
| All other presets | not enabled | | startup refuses `AGENT_MODE=on`. Enabling one is a preset field (`ProviderPreset.tools`) once its docs and a live test confirm it |

Model choice matters. Small local models often call tools when they
should not, or not when they should. The comparison below is how to find
out for a given model before turning the agent on.

## Provenance and observability

The assistant message's `generation_metadata.agent` records:
- `status`: `completed`, `completed_after_limit` or `fallback`, plus the
  `reason` for a fallback;
- the strategy summary (id, version, source, failure, tokens);
- `iterations`, `tool_calls` and `tools_called`;
- per-tool status, duration and count (never content);
- `model_calls`, the input and output tokens, `evidence_items` and
  `total_ms`;
- the strategies prompt key and checksum.

`knowledge` records the agent's own searches, where `retrieve` means the
model called `search_knowledge_base`. `memory` is
`{"mode": "agent_tools"}`. Logs: `agent.completed`, `agent.fallback`,
`agent.tool`, `agent.strategy_failed` / `agent.strategy_invalid`, and an
`agent` field on `turn.completed`. The `agent_strategies` version is in
the turn's version record.

## Evaluation

```bash
# Deterministic (no key): the guarantees with the agent in the turn
uv run python scripts/eval_golden.py --mode deterministic --agent
# Live: the same scenarios with and without the agent, then the comparison
uv run python scripts/eval_golden.py --mode live --split dev --pause 20
uv run python scripts/eval_golden.py --mode live --split dev --pause 20 --agent
uv run python scripts/eval_agent_compare.py --mode live --split dev
```

`eval_agent_compare.py` reports, side by side:
- guarantee, heuristic and reply pass rates, overall and per category;
- the scenarios whose result changed, and which checks failed;
- latency and model tokens per turn;
- the agent's strategies, tools and tool errors, plus its fallback and
  after-limit rates;
- **knowledge agreement**: how often the agent searched the knowledge
  base exactly when the scenario expects a lookup.

Deterministic baseline (2026-11-22, all 191 scenarios,
`eval/results/agent-compare-deterministic-all-20261122T143604.json`):

| | without agent | with agent |
|---|---|---|
| guarantee pass rate | 1.0 | 1.0 (new mandatory gate) |
| heuristic pass rate | 0.973 | 0.656 |
| knowledge agreement | 0.988 | 0.826 |

The heuristic drop is expected there, and says nothing about a real
model. The stub never calls a tool, so every scenario that expects a
knowledge lookup fails, and the stub always picks the first strategy.
What the deterministic run shows is that the agent path keeps every
guarantee: status codes under dependency failures, privacy probes,
memory deletion, citations that resolve, and safety statements.

**Not yet run:** the live comparison. It needs the reply model (the
Groq free tier, 200K tokens a day per model, or Ollama on your machine).
An agent turn costs more tokens than a deterministic one: the strategy
call plus one call per iteration, each resending the history and tool
results. Expect roughly 2-4 times the reply tokens on turns that use
tools.

## Threats specific to agent turns

See docs/threat-model.md, "Agent turns", for the full rows. In short:
- **Tool results are untrusted.** They arrive as tool-role messages in
  escaped blocks, and the prompt says to ignore instructions in them.
  Tests plant a closing tag and fake instructions in stored messages.
- **Cross-user reads are impossible by construction.** Identity is not
  an argument, and every query filters on the authenticated user
  (`tests/security/test_agent_tool_isolation.py`).
- **No write or side-effect tools.** The model cannot store memories,
  change state, browse, run code or query arbitrary tables.
- **Cost amplification is bounded** by the iteration, tool-call and
  deadline limits. Every turn's token use is recorded.
- **Behaviour is unverified live.** The model can choose a poor strategy
  or skip a useful lookup. A poor reply still goes through the same
  output checks, and a failure falls back.

## Limitations

- The live agent vs deterministic comparison has not been run yet (see
  Evaluation).
- Strategy selection costs one extra model call per turn, on top of the
  pre-fetch calls, which still run because the fallback needs them.
- History search is full-text, not semantic. It finds what was said in
  similar words, in English stemming.
- Tool calls run one at a time, even when the model asks for several.
- Only Groq and Ollama have tool calling enabled.
- The strategies and the tool guidance are first versions, written
  without live tuning. The prompts are yours to rewrite: add a new
  version of `agent_strategies` and lock it (`scripts/lock_prompts.py`).
