# Safety layer (Phase 5)

The safety layer is a set of controls outside the reply prompt (spec §12).
Every turn goes through them:

1. **Input analysis** decides whether the message needs special handling.
2. **Policy** picks a pathway.
3. **The reply is constrained** to that pathway.
4. **Output checks** run before the reply is stored.

No single model, score or prompt is trusted to make a reply safe. **This is
not a clinical system.** It has not been clinically validated, the texts the
person sees need review by a qualified person before any real use, and the
evaluation below is indicative only.

Code: `app/safety/` (taxonomy, patterns, classifier, analysis, policy,
resources, output checks, fallbacks, factory) and
`app/orchestration/safety_flow.py` (its part of a turn).

## Taxonomy (operational, not clinical)

Categories are listed most severe first. When several apply, the most severe
wins.

| Category | Meaning | Pathway | What the reply does |
|---|---|---|---|
| `imminent_danger` | Self-harm or harm to others that is planned, under way, or with means or time stated | `emergency` | Short and direct: emergency services now (listed contact), move away from means, get someone they trust to be with them, keep talking |
| `self_harm` | Thoughts of suicide or self-harm, a wish to be dead, recent self-harm | `crisis_support` | Warm. Glad they said it, they matter, asks gently whether they are safe now, offers a crisis line (listed) and someone they trust, keeps talking |
| `harm_to_others` | Intent or urge to seriously hurt someone | `harm_to_others` | Takes it seriously, helps with nothing harmful, encourages stepping away, crisis line or emergency services |
| `abuse` | The person or a child is being abused or is unsafe at home | `abuse_support` | Believes them, safety first, emergency contact if in danger now, listed support, no pressure to confront |
| `dangerous_request` | Asks for information that could seriously hurt someone (methods, doses, weapons) | `refuse_harmful` | Declines the details kindly, responds to what's underneath; crisis support if it's about themselves |
| `ambiguous_concern` | Could signal risk, unclear ("I can't do this anymore") | `safety_check_in` | Warm, then asks directly whether they're having thoughts of hurting themselves; no helpline list unless they say yes |
| `dependency` | Unhealthy reliance on the companion | `encourage_connection` | Stays warm and gently encourages other people and support; never pushes them away, never says it can replace them |
| `distress` | Upset without risk indicators | `normal` | The ordinary response policy |
| `none` | Ordinary conversation, including idioms, games, fiction, news | `normal` | The ordinary response policy |

**Any pathway other than `normal` overrides the ordinary strategy.** This
includes an explicit "I don't want advice": crisis guidance is never removed.
The companion prompt `companion_system@1.6.0` carries one fragment per
pathway. After the fragment, code renders the verified support contacts for
`SAFETY_REGION`, and the prompt says never to give any other number, website
or email.

## Input analysis

Analysis runs concurrently with state extraction and memory retrieval, and
it cannot fail the turn. It has three signals:

1. **Pattern screen** (`patterns.py`): deterministic, local, always on.
   - It normalises text first: Unicode confusables, leetspeak, spacing tricks,
     repeated letters and zero-width characters.
   - Rules are versioned, with ids, and are either *explicit* or *concern*.
   - Guards mask idioms ("this job is killing me lol", "kill a process",
     "dying to see", accidents, game commands). Fiction downgrades explicit
     hits to concern.
   - Imminence cues ("tonight", "pills here") count only together with an
     explicit harm hit in this message or the person's previous one.
2. **LLM classifier** (`classifier.py`).
   - **Policy prompt:** `prompts/safety_policy/1.0.0.toml`, in the
     gpt-oss-safeguard format.
   - **Input:** the latest message and up to four earlier ones, as escaped
     JSON marked untrusted.
   - **Output:** JSON `{category, rationale}`. The rationale is discarded
     unread, because it can quote the message.
   - **Default model:** Groq `openai/gpt-oss-safeguard-20b` at low effort.
     It is not the reply model, so it uses its own free-tier limits
     (3 requests and 2K tokens a minute, 200K tokens a day).
3. **Decision signal:** a calibrated `conversation_mode = safety` raises the
   category to at least `self_harm`. No decision model passes calibration
   yet (docs/decisions.md), so this never acts today.

### Combination (`analysis.py`)

- **Classifier works:** the category is the most severe of the classifier's
  label and the explicit pattern hits.
  - Explicit first-person statements are never downgraded.
  - Context-dependent rules (`CONTEXTUAL_RULES`) defer to a "no risk" label
    from the classifier. These are slang such as "kms", and self-injury words
    that also describe accidents or the past ("I used to self-harm at 15").
  - Concern-only hits don't override the classifier.
- **Classifier unavailable** (timeout, rate limit, invalid answer, error,
  disabled): **degraded mode**.
  - Explicit hits still route. Any concern hit gives `ambiguous_concern`.
  - With no hit at all, the turn takes the ordinary path and is recorded as
    **unassessed** (`assessed = false`), never as safe.
  - The reply prompt's danger guidance is always present, and the output
    checks run in their strict form on unassessed turns.

## Output checks and bounded recovery (`output.py`, `fallbacks.py`)

Deterministic checks run on every reply:

| Code | Catches |
|---|---|
| `dangerous_details` | Methods, doses or means. Strict on safety pathways and unassessed turns |
| `missing_required` | A crisis or emergency reply without crisis help and someone they trust |
| `claims_human_or_clinician` | "I'm a therapist", "as a doctor", "I'm not an AI" |
| `secrecy_or_isolation` | "keep this between us", "you don't need anyone else" |
| `clinical_label` | "you have depression", "sounds like you have BPD" |
| `manipulation` | "if you leave I'll be sad" |
| `prompt_leak` | Internal markers, or any 12-word run of the system prompt |
| `outdated_resource` | Retired services (e.g. Samaritans' email, closed 30 Sep 2026; India's KIRAN line) |
| `unlisted_contact` | Any phone number or email address not in the region's verified list, on every pathway (a live knowledge run invented a service number on an ordinary turn); on safety pathways also any unlisted link |

On safety pathways a **semantic check** (`prompts/safety_output/1.0.0.toml`,
same model as the classifier) also asks whether the reply gives harmful
instructions or fails the pathway. It runs only when the deterministic
checks pass, and its own failure never blocks the reply.

**Recovery:**

1. A failing reply is regenerated **once**, with a note naming the problems
   (codes only; the rejected reply is never quoted back).
2. If the second reply also fails, or can't be generated within the turn
   deadline, the turn stores the pathway's **fixed fallback**. On safety
   pathways the fallback includes the rendered contacts; otherwise it is a
   generic "Sorry, I couldn't put that reply together properly".

A failed regeneration never fails the turn. The fallback texts are
application copy for people who may be in crisis, and they need review by a
qualified person.

## Crisis resources (`resources_v1.toml`)

- **Coverage:** India, the US and the UK, plus a generic set with no numbers
  (`SAFETY_REGION=""`).
- **Sources:** every entry comes from an official page or document and
  records its source URL and the date it was checked (2026-10-13). For
  example, India has 112 and Tele-MANAS 14416; the US 911, 988 and Crisis
  Text Line; the UK 999, Samaritans 116 123, NHS 111 and Shout 85258.
- **Not used:** Find A Helpline's data; its terms forbid it.
- **Freshness:** a unit test fails after the file's `review_due` date
  (2027-01-11), so the list cannot silently go stale.

## Records and privacy

Table `turn_safety` (migration 0005) holds one row per completed turn:

- category, pathway, `assessed`, `degraded`;
- classifier status, model and label;
- pattern rule ids, policy and pattern versions, region;
- output violation codes, semantic check status, recovery, latency.

**It holds no message text and no rationale.** Rows are deleted with their
conversation. Logs carry the same labels only.

The API response of `POST /v1/conversations/{id}/messages` has a `safety`
object when a safety pathway applied: the pathway and the verified contacts,
so a client can show them. It is also returned on replay.

## Configuration

| Setting | Default | |
|---|---|---|
| `SAFETY_ENABLED` | `true` | Refused as `false` in production |
| `SAFETY_CLASSIFIER_PROVIDER` | `auto` | `auto`: Groq when `GROQ_API_KEY` is set, otherwise off (degraded mode). Or `off`, or any provider preset |
| `SAFETY_CLASSIFIER_MODEL` | `openai/gpt-oss-safeguard-20b` | Groq preview model. `openai/gpt-oss-20b` also works (30 requests a minute) |
| `SAFETY_CLASSIFIER_EFFORT`, `..._MAX_OUTPUT_TOKENS`, `..._TIMEOUT_SECONDS` | `low`, 300, 4 s | |
| `SAFETY_REGION` | `""` | `IN`, `US`, `GB`, or empty for the generic set. Unknown codes fail at startup |
| `SAFETY_OUTPUT_LLM_CHECK` | `true` | The semantic output check on safety pathways |
| `SAFETY_POLICY_VERSION`, `SAFETY_OUTPUT_POLICY_VERSION` | `1.0.0` | Locked prompts |

**Cost per turn on the free tier:**

- **Classifier:** one call per message to the classifier model, about
  750-850 tokens (measured).
- **Semantic check:** one more classifier-model call, only on safety
  pathways.
- **Regeneration:** one extra reply-model call, only when a reply fails a
  check.

At 3 requests a minute, a burst of more than three messages a minute puts
the extra messages in degraded mode, which is recorded.

## Evaluation

The data is in `eval/datasets/safety/v1/` (see its DATASET_CARD.md):

- 100 authored hard negatives and lower-category items, AI-labelled by a
  majority of three;
- the 13 danger rows of the decision set;
- XSTest, SimpleSafetyTests, OR-Bench and Do-Not-Answer, downloaded and
  pinned by `scripts/import_safety_benchmarks.py`.

```
uv run python scripts/import_safety_benchmarks.py
uv run python scripts/eval_safety.py --mode patterns                         # free, every item
uv run python scripts/eval_safety.py --mode classifier --split dev --pause 21
```

Missed risk and false positives are reported separately, per source and per
category.

## Results (dev split, 2026-10-15)

The full layer (patterns plus the `openai/gpt-oss-safeguard-20b` classifier,
policy `safety_policy@1.0.0`) on the 172 dev items, against patterns alone
(degraded mode) on the same items.

**Files:** `eval/results/safety-classifier-dev-20261015T162330.json` and
`safety-patterns-all-20261015T162345.json`. Ids and labels only, no text.

| | Patterns only | Patterns + classifier |
|---|---|---|
| Risk items missed (63 items, sent down no safety pathway) | 73.0% | **23.8%** |
| Clear items sent to a crisis or refusal pathway (93) | 1.1% | **10.8%** |
| Lower-category items (ambiguous concern, dependency) given a safety pathway (16) | 6.2% | 37.5% |
| Category exactly right | 46.5% | 78.2% |

**What was missed (15 risk items).**

- 11 were requests for harmful information aimed at others, from
  SimpleSafetyTests, XSTest's unsafe prompts and OR-Bench-Toxic, that the
  classifier labelled `none`.
- 4 had an invalid classifier answer (see Degraded mode below), and the
  patterns did not catch them.
- 2 were decision-set danger rows (one labelled `distress`, one `none`).
- Explicit first-person self-harm statements were not among the misses.
- For these requests the reply model's own refusal training and the
  `dangerous_details` output check still apply. Neither is measured here.

**False positives (10 of 93).**

- 8 went to `refuse_harmful`, mostly OR-Bench-Hard prompts. That benchmark is
  built to sound harmful while being benign; 6 of its 20 items went to a
  safety pathway.
- 1 went to `crisis_support` and 1 to `harm_to_others`.
- On XSTest's safe prompts the false-positive rate was 4% (1 of 25).

**Degraded mode.** 8 of 172 classifier answers (4.7%) were not valid JSON
(the model is a preview with best-effort JSON). Those turns ran in degraded
mode; 6 of them had no pattern hit and were recorded as unassessed.

**Latency.** The classifier call took 0.66 s at the median, 0.85 s at the
90th percentile and 2.7 s at most, within the 4 s timeout. It runs alongside
state extraction and memory retrieval, so it adds little to a turn.

**Reading these numbers.**

- They come from benchmark and AI-authored prompts, not real conversations.
- Immediate danger and abuse are almost absent from the data (see Limits).
- The test split has not been run. It is kept for a later check, after which
  nothing is tuned.

## Limits

- **Barely measured cases.** People in immediate danger, and people
  disclosing abuse, appear in very few evaluation items. The platform
  stopped an AI author from writing such messages, and no public dataset
  covers them. Human-reviewed examples are the most important gap.
- **Patterns don't work in degraded mode for indirect wording.** They catch
  explicit statements and miss most indirect or adversarial phrasing; the
  classifier is the main detector.
- **Preview model.** The classifier model is a Groq preview and may be
  withdrawn. `SAFETY_CLASSIFIER_MODEL=openai/gpt-oss-20b` is the fallback
  (re-evaluate first).
- **English only**, with some Hinglish in the patterns.
- **Some false positives are accepted.** In degraded mode, context-dependent
  hits ("kms" as a joke) still go to crisis support. With the classifier
  working they defer to it.
