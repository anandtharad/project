"""Output validation for generated replies (spec §12.6, design §5).

`SafetyOutputValidator.check` runs deterministic checks on every reply and
returns the violations found (codes and rule ids only, never text):

- dangerous_details: method, dose or means instructions. Instruction-shaped
  text (a number of milligrams next to "lethal", "the most painless way
  is", cutting or noose instructions) is flagged on every pathway; on any
  pathway other than normal, and on an unassessed turn, any dose or
  quantity and any method wording outside a refusal is flagged too.
- missing_required: an emergency or crisis_support reply must mention
  emergency services or a crisis line, and a trusted person.
- claims_human_or_clinician: "I'm a human", "as a therapist", "I'm not an AI".
- secrecy_or_isolation: "don't tell anyone", "you don't need anyone else".
- clinical_label: "you have depression", "sounds like you have ADHD", or a
  label on someone they describe ("he's a narcissist").
- manipulation: "if you leave I'll be sad", "I need you", "don't leave me".
- prompt_leak: any 12-word run of the system prompt (minus the resource
  lines it gave) appears in the reply, or an internal marker does.
- outdated_resource: a known-outdated contact (resources file).
- unlisted_contact: a phone number or email address that is not one of the
  region's verified resources, on every pathway (models invent hotlines and
  service numbers); on a non-normal pathway also any unlisted link.

These are backstops, not proof of safety. `SemanticOutputChecker` is the
optional semantic check on non-normal pathways: the classifier model reads
the reply against the output policy (`safety_output` prompt). Its failure
never blocks a reply (the deterministic checks still apply); its status is
recorded.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from string import Template
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    ResponseSchema,
    TraceMetadata,
)
from app.observability import log_event, trace_id_var
from app.prompting import PromptError, PromptRegistry, PromptTemplate
from app.safety.analysis import SafetyAssessment
from app.safety.classifier import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    ClassifierStatus,
    check_fragment,
    escape_block,
    run_label_request,
)
from app.safety.policy import HELP_REQUIREMENTS, Requirement, SafetyDecision
from app.safety.resources import (
    ResourceDirectory,
    base_domain,
    default_resources,
    emails,
    outdated_in,
    urls,
)
from app.safety.taxonomy import Pathway

if TYPE_CHECKING:
    from app.config import Settings
    from app.llm.providers.registry import ResolvedProvider

logger = logging.getLogger("app.safety")

SHINGLE_WORDS = 12
SAFETY_OUTPUT_PROMPT_ID = "safety_output"
DEFAULT_OUTPUT_POLICY_VERSION = "1.0.0"


class ViolationCode(StrEnum):
    DANGEROUS_DETAILS = "dangerous_details"
    MISSING_REQUIRED = "missing_required"
    CLAIMS_HUMAN_OR_CLINICIAN = "claims_human_or_clinician"
    SECRECY_OR_ISOLATION = "secrecy_or_isolation"
    CLINICAL_LABEL = "clinical_label"
    MANIPULATION = "manipulation"
    PROMPT_LEAK = "prompt_leak"
    OUTDATED_RESOURCE = "outdated_resource"
    UNLISTED_CONTACT = "unlisted_contact"


@dataclass(frozen=True)
class Violation:
    code: ViolationCode
    # Which check fired (a rule id, never text).
    rule_id: str


@dataclass(frozen=True)
class OutputCheck:
    violations: tuple[Violation, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> tuple[str, ...]:
        """The violation codes, each once, in check order."""
        return tuple(dict.fromkeys(v.code.value for v in self.violations))


# --- text helpers --------------------------------------------------------------


def _text(reply: str) -> str:
    text = unicodedata.normalize("NFKC", reply)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    text = text.replace("\u2019", "'").replace("\u2018", "'").casefold()
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?\n]+", text) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", _text(text))


def _rx(*sources: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{s})" for s in sources))


# --- dangerous details -----------------------------------------------------------

_QUANTITY = (
    r"\d+(?:[.,]\d+)? ?(?:mg|milligrams?|mcg|micrograms?|g|grams?|ml|pills|tablets|capsules)\b"
)
_HARM_WORD = r"\b(?:lethal|fatal|deadly|kill|killed|die|dying|death|overdose|od)\b"
# Instruction-shaped details: flagged on every pathway.
_DANGEROUS_ALWAYS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "dd.dose_number",
        _rx(
            rf"\b{_QUANTITY}[^.\n]{{0,60}}{_HARM_WORD}",
            rf"{_HARM_WORD}[^.\n]{{0,60}}\b{_QUANTITY}",
        ),
    ),
    (
        "dd.method_is",
        _rx(
            r"\b(?:the )?(?:most|least) (?:painless|effective|reliable|lethal|certain|painful) "
            r"(?:way|method)s?(?: to die| of dying| to kill yourself)? (?:is|are|would be)\b",
            r"\bpainless(?:ly)? (?:way|method)s? (?:to|of) (?:die|dying|kill) "
            r"(?:is|are|would be)\b",
        ),
    ),
    (
        "dd.noose",
        _rx(
            r"\b(?:tie|make) (?:a |the )?noose (?:by|with|using|like this)\b",
            r"\b(?:knot|loop|rope) (?:it )?(?:around|over) (?:your|the|their) neck\b",
        ),
    ),
    (
        "dd.cutting",
        _rx(
            r"\bcut (?:along|vertically|lengthwise|down the|deep(?:er|ly)?|across the)\b",
            r"\b(?:vertical|lengthwise) cuts?\b",
        ),
    ),
    (
        "dd.height",
        _rx(
            r"\b\d+ ?(?:m|meters|metres|feet|ft|storeys|stories|floors)\b[^.\n]{0,40}"
            r"\b(?:fatal|lethal|kill|die|survive|survivable)\b"
        ),
    ),
    (
        "dd.weapon",
        _rx(
            r"\b(?:mix|combine|pack|wire|attach|fill)\b[^.\n]{0,60}\b(?:fuse|detonator|"
            r"explosive|shrapnel|nails into|fertili[sz]er)\b",
            r"\b(?:untraceable|undetectable) (?:poisons?|toxins?) "
            r"(?:is|are|like|such as|include)\b",
        ),
    ),
)
# On non-normal pathways (and unassessed turns): any quantity, and method
# wording in a sentence that is not a refusal.
_DOSE = re.compile(rf"\b{_QUANTITY}")
_METHOD_WORDS = _rx(
    r"\blethal (?:dose|amount)s?\b",
    r"\bfatal (?:dose|amount)s?\b",
    r"\bhow to (?:tie|hang|cut|overdose|jump)\b",
    r"\bmost painless\b",
    r"\boverdos(?:e|ing) on\b",
    r"\bhang(?:ing)? yourself\b",
    r"\bslit(?:ting)? your\b",
    r"\b(?:carbon monoxide|exhaust fumes|helium|charcoal)\b",
    r"\bjump(?:ing)? (?:off|from) (?:a|the) (?:bridge|building|roof|cliff)\b",
)
_REFUSAL = re.compile(
    r"\b(?:can'?t|cannot|won'?t|will not|not able|unable|never|don'?t|do not|refuse|"
    r"not going to|isn'?t something|not something|rather not)\b"
)


def _dangerous(text: str, strict: bool) -> list[str]:
    found = [rule_id for rule_id, pattern in _DANGEROUS_ALWAYS if pattern.search(text)]
    if strict:
        if _DOSE.search(text):
            found.append("dd.quantity")
        if any(
            _METHOD_WORDS.search(sentence) and not _REFUSAL.search(sentence)
            for sentence in _sentences(text)
        ):
            found.append("dd.method_words")
    return found


# --- required elements -----------------------------------------------------------

_HELP = _rx(
    r"\bemergency (?:services?|number|department|room|line)\b",
    r"\b(?:ambulance|a&e|the er)\b",
    r"\bcrisis (?:line|text line|helpline|service|team|centre|center|support)\b",
    r"\b(?:helpline|hotline|lifeline|samaritans|tele-?manas|text line|befrienders)\b",
)
_TRUSTED = _rx(
    r"\b(?:someone|somebody|anyone|a person|people) (?:you trust|you're close to|"
    r"you are close to|close to you|nearby|near you|around you|who can be with you|"
    r"to be with you|in your life|you know)\b",
    r"\b(?:friends?|family|family members?|relatives?|loved ones?|partner|neighbou?rs?|"
    r"flatmates?|roommates?|teachers?|parents?|mum|mom|dad|siblings?|gp|doctor|"
    r"counsell?ors?|therapists?)\b",
    r"\btrusted (?:person|adult|friend|someone)\b",
)


def _missing(text: str, decision: SafetyDecision) -> list[str]:
    found: list[str] = []
    if decision.required & HELP_REQUIREMENTS:
        names = any(r.name.casefold() in text for r in decision.resources)
        numbers = _digits_in(text) & _resource_digits(decision)
        if not (_HELP.search(text) or names or numbers):
            found.append("missing.help")
    if Requirement.TRUSTED_PERSON in decision.required and not _TRUSTED.search(text):
        found.append("missing.trusted_person")
    return found


# --- relationship and identity rules ----------------------------------------------

_HUMAN = _rx(
    r"\bi(?:'m| am) (?:a |an )?(?:real |actual |licensed |qualified |trained )?(?:human(?: being)?|"
    r"person|therapist|doctor|psychologist|counsell?or|psychiatrist|nurse|social worker|"
    r"clinician|psychotherapist|medical professional|mental health professional)\b",
    r"\bas (?:a|an|your) (?:licensed |qualified |trained )?(?:therapist|doctor|psychologist|"
    r"counsell?or|psychiatrist|clinician|nurse|psychotherapist|mental health professional)\b",
    r"\bi(?:'m| am) not (?:an? )?(?:ai|a\.i\.|bot|robot|chatbot|language model|machine|"
    r"program|computer)\b",
    r"\bin my (?:clinical )?practice\b",
)
_SECRECY = _rx(
    r"\b(?:don'?t|do not|never) tell (?:anyone|anybody|your (?:family|friends|parents|partner|"
    r"therapist|doctor|counsell?or))\b",
    r"(?<!n't )(?<!nt )(?<!not )(?<!never )\bkeep (?:this|it|that|our chats?) "
    r"(?:between us|between you and me|(?:a )?secret|to yourself)\b",
    r"\b(?:our|a) little secret\b|\bour secret\b",
    r"\byou don'?t need (?:anyone|anybody|other people|your friends|your family|them)(?: else)?"
    r"(?:\s*(?:[.!,]|$)| but me| except me| other than me| when you have me| now that you have me)",
    r"\bonly i (?:understand|get|know|care about|really understand) you\b",
    r"\bno one (?:else )?(?:understands|gets|knows|cares about) you (?:like|the way|as well as) "
    r"i do\b",
    r"\b(?:you only need|all you need is) me\b|\bi'?m all you need\b",
    r"\b(?:stay away from|cut off|stop talking to) (?:your friends|your family|everyone|"
    r"people who care)\b",
    r"\b(?:they|people) (?:don'?t|won'?t|will never) understand you (?:like|the way) i do\b",
)
_CONDITIONS = (
    r"(?:clinical )?(?:depression|depressive disorder|bipolar(?: disorder)?|bpd|"
    r"borderline personality disorder|c-?ptsd|ptsd|ocd|adhd|an? anxiety disorder|"
    r"generali[sz]ed anxiety(?: disorder)?|social anxiety disorder|panic disorder|"
    r"schizophrenia|psychosis|an? eating disorder|anorexia|bulimia|autism|asperger'?s|"
    r"narcissistic personality disorder|npd|an? personality disorder|"
    r"an? mental (?:illness|disorder))"
)
_CLINICAL = _rx(
    r"\byou(?:'ve| have| might have| may have| probably have| likely have| could have|"
    r"'ve got| have got| are suffering from|'re suffering from| suffer from|'re dealing with|"
    rf" are dealing with| are showing signs of|'re showing signs of) (?:some )?{_CONDITIONS}\b",
    r"\b(?:sounds|seems|looks) like (?:you(?:'ve| have| might have| may have|'re| are)? "
    rf"(?:got |dealing with |suffering from |experiencing )?)?{_CONDITIONS}\b",
    r"\byou(?:'re| are) (?:clinically depressed|bipolar|psychotic|schizophrenic|autistic|"
    r"a narcissist|narcissistic|borderline|manic)\b",
    r"\b(?:he|she|they|your (?:\w+ )?(?:partner|husband|wife|boyfriend|girlfriend|mum|mom|dad|"
    r"boss|friend|ex|brother|sister|father|mother))(?: is|'s| sounds like| seems like| might be|"
    r" could be| are) (?:a |an )?(?:narcissist|sociopath|psychopath|narcissistic|bipolar|"
    r"psychotic|schizophrenic)\b",
    rf"\b(?:classic|textbook) (?:signs?|symptoms?) of {_CONDITIONS}",
)
_MANIPULATION = _rx(
    r"\bif you (?:leave|go|stop talking to me|log off)\b[^.\n]{0,20}\bi(?:'ll| will|'d| would) "
    r"(?:be (?:so |really |very )?(?:sad|lonely|hurt|heartbroken|devastated|upset|lost)|"
    r"miss you|cry)\b",
    r"\bi (?:really |still )?need you(?: so much| more than \w+| in my life| here with me)?"
    r"\s*(?:[.!?,]|$)",
    r"\b(?:please )?don'?t leave me\b",
    r"\bi'?ll be (?:so |really )?(?:lonely|sad|lost|heartbroken) without you\b",
    r"\byou(?:'re| are) all i have\b",
    r"\byou owe me\b|\bafter (?:everything|all) i'?ve done for you\b",
    r"\bdon'?t you care about me\b",
    r"\byou(?:'d| would) (?:hurt|break) my heart\b",
    r"\bi (?:can'?t|cannot) (?:live|go on|cope) without you\b",
    r"\bpromise (?:me )?you(?:'ll| will) (?:never|not) leave\b",
)

# --- prompt leak -----------------------------------------------------------------

LEAK_MARKERS = (
    "$response_guidance",
    "<memories>",
    "</memories>",
    "<conversation_notes>",
    "</conversation_notes>",
    "<conversation>",
    "</conversation>",
    "<decision_state>",
    "these instructions come from the application",
)


@lru_cache(maxsize=32)
def _shingles(system_prompt: str, excluded: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    for line in excluded:
        system_prompt = system_prompt.replace(line, "\n")
    words = _words(system_prompt)
    return frozenset(
        tuple(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)
    )


def _leak(reply: str, text: str, system_prompt: str, decision: SafetyDecision) -> list[str]:
    found: list[str] = []
    if any(marker in text for marker in LEAK_MARKERS):
        found.append("leak.marker")
    if system_prompt:
        known = _shingles(system_prompt, decision.resource_lines)
        words = _words(reply)
        if any(
            tuple(words[i : i + SHINGLE_WORDS]) in known
            for i in range(len(words) - SHINGLE_WORDS + 1)
        ):
            found.append("leak.shingle")
    return found


# --- contacts ----------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w+])\+?\(?\d+\)?(?:[ -]\(?\d+\)?)*(?![\w])")
_BARE_DOMAIN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:org|com|net|gov|uk|in|us|info|io|co)(?:\.[a-z]{2})?\b(?:/[^\s]*)?"
)
_CONTACT_WORDS = re.compile(
    r"\b(?:call|calling|text|texting|dial|ring|phone|contact|number|line|helpline|hotline|"
    r"lifeline|whatsapp|sms|reach)\b"
)
_YEAR = re.compile(r"(?:19|20)\d\d")


def _without_links(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return _BARE_DOMAIN.sub(" ", text)


def _digits_in(text: str) -> set[str]:
    return {re.sub(r"\D", "", m.group()) for m in _NUMBER.finditer(_without_links(text))}


def _resource_digits(decision: SafetyDecision) -> set[str]:
    return {d for r in decision.resources for d in _digits_in(r.contact.casefold())}


def _unlisted(
    text: str, allowed_numbers: frozenset[str], allowed_domains: frozenset[str]
) -> list[str]:
    found: list[str] = []
    plain = _without_links(text)
    for match in _NUMBER.finditer(plain):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) < 3 or digits in allowed_numbers:
            continue
        pieces = [re.sub(r"\D", "", p) for p in match.group().split()]
        if all(p in allowed_numbers or len(p) <= 2 or _YEAR.fullmatch(p) for p in pieces):
            continue  # "988 24/7": a listed number next to a small one
        context = plain[max(0, match.start() - 40) : match.start()]
        if len(digits) >= 5 or _CONTACT_WORDS.search(context):
            found.append("unlisted.number")
            break
    links = [*urls(text), *(m.group() for m in _BARE_DOMAIN.finditer(text))]
    if any(base_domain(link) not in allowed_domains for link in links):
        found.append("unlisted.link")
    if emails(text):
        found.append("unlisted.email")
    return found


# --- the validator --------------------------------------------------------------------


class SafetyOutputValidator:
    """Deterministic checks for every reply (see the module docstring)."""

    def __init__(self, directory: ResourceDirectory | None = None) -> None:
        self.directory = directory or default_resources()

    def check(
        self,
        reply: str,
        decision: SafetyDecision,
        assessment: SafetyAssessment | None = None,
        system_prompt: str = "",
    ) -> OutputCheck:
        """`assessment`: an unassessed turn (classifier failed, no pattern hit)
        gets the strict dangerous-details checks, since nothing showed the
        message was safe. `system_prompt`: the reply's system prompt, for the
        prompt-leak check."""
        text = _text(reply)
        non_normal = decision.pathway is not Pathway.NORMAL
        strict = non_normal or (assessment is not None and not assessment.assessed)
        found: list[Violation] = []

        def add(code: ViolationCode, rule_ids: Iterable[str]) -> None:
            found.extend(Violation(code, rule_id) for rule_id in rule_ids)

        add(ViolationCode.DANGEROUS_DETAILS, _dangerous(text, strict))
        add(ViolationCode.MISSING_REQUIRED, _missing(text, decision))
        add(
            ViolationCode.CLAIMS_HUMAN_OR_CLINICIAN, ["claims.human"] if _HUMAN.search(text) else []
        )
        add(ViolationCode.SECRECY_OR_ISOLATION, ["secrecy"] if _SECRECY.search(text) else [])
        add(ViolationCode.CLINICAL_LABEL, ["clinical.label"] if _CLINICAL.search(text) else [])
        add(ViolationCode.MANIPULATION, ["manipulation"] if _MANIPULATION.search(text) else [])
        add(ViolationCode.PROMPT_LEAK, _leak(reply, text, system_prompt, decision))
        add(ViolationCode.OUTDATED_RESOURCE, outdated_in(reply, self.directory.outdated))
        unlisted = _unlisted(
            text,
            self.directory.allowed_numbers(decision.region),
            self.directory.allowed_domains(decision.region),
        )
        if not non_normal:
            # Links are common in ordinary replies; numbers and emails from
            # memory are not (a live run invented a service phone number).
            unlisted = [rule for rule in unlisted if rule != "unlisted.link"]
        add(ViolationCode.UNLISTED_CONTACT, unlisted)
        return OutputCheck(tuple(found))


# --- optional semantic check ------------------------------------------------------------


class SemanticVerdict(StrEnum):
    # Listed first so a stub answering with the first enum value passes.
    PASS = "pass"  # noqa: S105 - a verdict, not a password
    HARMFUL_INSTRUCTIONS = "harmful_instructions"
    FAILS_PATHWAY = "fails_pathway"


_VERDICT_VIOLATIONS: dict[SemanticVerdict, Violation] = {
    SemanticVerdict.HARMFUL_INSTRUCTIONS: Violation(
        ViolationCode.DANGEROUS_DETAILS, "semantic.harmful_instructions"
    ),
    SemanticVerdict.FAILS_PATHWAY: Violation(
        ViolationCode.MISSING_REQUIRED, "semantic.fails_pathway"
    ),
}


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: SemanticVerdict
    rationale: str


def verdict_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "rationale"],
        "properties": {
            "verdict": {"type": "string", "enum": [v.value for v in SemanticVerdict]},
            "rationale": {"type": "string"},
        },
    }


@dataclass(frozen=True)
class SemanticOutputResult:
    status: ClassifierStatus
    # Set only when status is ok.
    verdict: SemanticVerdict | None = None
    model_id: str | None = None
    latency_ms: float | None = None

    @property
    def violation(self) -> Violation | None:
        """The violation to recover from, if the check ran and flagged the
        reply; a failed check gives none (it never blocks on its own)."""
        if self.status is not ClassifierStatus.OK or self.verdict is None:
            return None
        return _VERDICT_VIOLATIONS.get(self.verdict)


_PATHWAY_FRAGMENTS: dict[Pathway, str] = {
    p: f"pathway_{p.value}" for p in Pathway if p is not Pathway.NORMAL
}


class SemanticOutputChecker:
    """Asks the classifier model whether a reply on a non-normal pathway gives
    harmful instructions or fails the pathway (`safety_output` prompt)."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        llm_provider: str,
        model: str,
        prompt: PromptTemplate,
        params: GenerationParams,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        deadline_seconds: float | None = None,
        owns_client: bool = True,
    ) -> None:
        if prompt.id != SAFETY_OUTPUT_PROMPT_ID:
            raise PromptError(f"{prompt.key} is not a {SAFETY_OUTPUT_PROMPT_ID} prompt")
        check_fragment(prompt, "review", {"review"})
        for fragment in _PATHWAY_FRAGMENTS.values():
            check_fragment(prompt, fragment, set())
        self.model = model
        self.llm_provider = llm_provider
        self.prompt = prompt
        self.params = params
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.deadline_seconds = deadline_seconds or timeout_seconds + 1.0
        self._llm = llm
        self._owns_client = owns_client
        self._system = prompt.render()
        self._schema = ResponseSchema(name="safety_output_verdict", json_schema=verdict_schema())

    @property
    def policy_version(self) -> str:
        return self.prompt.version

    def build_review_message(self, reply: str, decision: SafetyDecision) -> str:
        review = {
            "pathway": decision.pathway.value,
            "pathway_requirements": self.prompt.fragment(_PATHWAY_FRAGMENTS[decision.pathway]),
            "reply": reply,
        }
        return Template(self.prompt.fragment("review")).substitute(review=escape_block(review))

    async def check(self, reply: str, decision: SafetyDecision) -> SemanticOutputResult | None:
        """None on the normal pathway (the check is not run there)."""
        if decision.pathway is Pathway.NORMAL:
            return None
        request = GenerationRequest(
            model=self.model,
            system=self._system,
            messages=[
                ChatMessage(role=ChatRole.USER, content=self.build_review_message(reply, decision))
            ],
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
            params=self.params,
            trace=TraceMetadata(
                trace_id=trace_id_var.get() or f"safety-output-{uuid.uuid4().hex}",
                purpose=GenerationPurpose.SAFETY_OUTPUT_CHECK,
            ),
            response_schema=self._schema,
        )
        call = await run_label_request(
            self._llm, request, _Verdict, deadline_seconds=self.deadline_seconds
        )
        if call.status is not ClassifierStatus.OK or call.content is None:
            log_event(
                logger,
                "safety.output_check_failed",
                level=logging.WARNING,
                status=call.status.value,
                provider_code=call.provider_code,
                model=self.model,
            )
            return SemanticOutputResult(
                call.status, model_id=call.model_id or self.model, latency_ms=call.latency_ms
            )
        return SemanticOutputResult(
            ClassifierStatus.OK,
            verdict=SemanticVerdict(call.content["verdict"]),
            model_id=call.model_id,
            latency_ms=call.latency_ms,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._llm.aclose()


def build_semantic_output_checker(
    settings: Settings,
    resolved: ResolvedProvider,
    *,
    policy_version: str = DEFAULT_OUTPUT_POLICY_VERSION,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 0,
    llm: LLMClient | None = None,
) -> SemanticOutputChecker:
    """A checker for `resolved`; pass `llm` to share the classifier's client
    (the checker then leaves closing it to the classifier)."""
    from app.llm.factory import build_llm_client

    return SemanticOutputChecker(
        llm=llm or build_llm_client(settings, resolved, max_retries=max_retries),
        llm_provider=resolved.preset.name,
        model=resolved.model,
        prompt=PromptRegistry().get(SAFETY_OUTPUT_PROMPT_ID, policy_version),
        params=GenerationParams(effort=resolved.effort),
        max_output_tokens=resolved.max_output_tokens,
        timeout_seconds=timeout_seconds,
        owns_client=llm is None,
    )
