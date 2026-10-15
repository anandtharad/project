"""When to look things up (spec §11.7): deterministic, conservative routing.

Retrieval runs only for explicit information requests about the topics the
corpus covers. Casual conversation and personal reflection never trigger
it, so a companion chat does not turn into a list of facts (Phase 4
acceptance: no unnecessary retrieval). The kinds:

- casual:            greetings, small talk, anything without a request
- reflection:        the person talking about themselves, others or their
                     feelings, including rhetorical questions ("what's the
                     point", "why do I always ruin things")
- factual:           a general question on a covered topic ("what is sleep
                     hygiene?", "any tips for falling asleep?")
- evidence:          asks for evidence or sources on a topic ("is there any
                     research on journaling?", "is that a myth?")
- source_followup:   asks where the last answer came from ("source?"); the
                     query is the person's previous message
- current_info:      asks for the latest or current information; the
                     library is offline, so the reply says the passages may
                     not be current (no web browsing in this phase)

Never retrieved: safety pathways (the engine drops results there), and
messages that say they don't want advice. A decision model is not used:
none passes calibration yet (docs/decisions.md), and these patterns are
measured on their own labelled set (eval/datasets/rag/v1/routing.jsonl).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.orchestration.policy import explicit_request

RouteKind = Literal[
    "casual", "reflection", "factual", "evidence", "source_followup", "current_info"
]


@dataclass(frozen=True)
class Route:
    kind: RouteKind
    retrieve: bool
    query: str
    reason: str


# Topics the corpus covers. A request about anything else is not looked up.
_TOPIC = re.compile(
    r"\b(?:"
    r"sleep\w*|asleep|insomnia|nap(?:s|ping)?|bed ?time|before bed|wind[- ]?down|"
    r"caffeine|coffee|screens?|screen time|night ?mares?|tired(?:ness)?|fatigue|"
    r"stress\w*|anxi\w+|panic\w*|worr(?:y|ies|ying)|burn[- ]?out|overwhelm\w*|"
    r"mindful\w*|meditat\w*|breath\w*|relax\w*|grounding|"
    r"journal\w*|exercis\w*|physical activity|walking|yoga|"
    r"mood|low mood|depress\w*|emotion\w*|"
    r"self[- ]care|wellbeing|well-being|mental health|resilien\w*|gratitude|"
    r"cbt|cognitive behaviou?ral|therapy|counsell?ing|"
    r"procrastinat\w*|motivat\w*|habits?|routines?|"
    r"lonel\w*|isolat\w*|grie(?:f|ving|ve)|bereave\w*|"
    r"communicat\w*|assertive\w*|boundar(?:y|ies)|conflicts?|arguments?|argu(?:e|ing)|"
    r"fights?|fighting|relationships?|partner|"
    r"self[- ]esteem|confidence|rumina\w*|overthink\w*|"
    r"diet|nutrition|alcohol|sugar|protein|vitamins?|vegetarian|vegan|eating|"
    r"exams?|revis(?:e|ing|ion)|studying|viva|focus\w*|concentrat\w*|scrolling|"
    r"small talk|social\w*|conversations?|awkward|shy(?:ness)?"
    r")\b",
    re.IGNORECASE,
)

# Rhetorical or personal forms that look like questions but ask for no
# information. Checked first.
_REFLECTIVE = re.compile(
    r"\bwhat(?:'s| is) wrong with (?:me|him|her|them|us)\b"
    r"|\bwhat(?:'s| is) the point\b"
    r"|\bwhat does it (?:even )?mean (?:that|when) (?:i|he|she|they|we|my)\b"
    r"|\bwhat does it even mean\b"
    r"|\bexplain (?:to me )?why (?:he|she|they|i|you|we|my)\b"
    r"|\bcan you explain why (?:i|he|she|they|we|my)\b"
    r"|\btell me about (?:it|your|yourself)\b"
    r"|\bhow (?:does|do|could|can) (?:he|she|they|people) (?:not|even)\b"
    r"|\bhow do you even\b"
    r"|\bhow (?:much )?longer can i\b"
    r"|\bhow long does this\b"
    # About the companion ("is it true that you care"), not generic "you".
    r"|\bis it true that you (?:really |actually )?(?:care|love|like|are|mean|think|feel|"
    r"remember|understand|listen|read)\b"
    r"|\bwhat are we\b"
    r"|\bwhy do i (?:always|keep|feel)\b"
    r"|\bis there (?:any|a|even a) point\b"
    r"|\bhow many times\b"
    r"|\bis it (?:normal|ok|okay) (?:to|that) (?:feel|be|i|i'm)\b.*\b(?:this|so|after)\b",
    re.IGNORECASE,
)

_SOURCE_FOLLOWUP = re.compile(
    r"^\W*(?:sources?|citations?|references?|links?)\W*(?:please|pls)?\W*$"
    r"|\bwhere (?:did|do) you (?:get|read|find|hear) (?:that|this|it)\b"
    r"|\bwhat(?:'s| is| are) (?:your|the) sources?\b"
    r"|\b(?:can|could) you (?:cite|link|give me) (?:a |the |your )?(?:source|link|reference)s?\b"
    r"|\bsource for (?:that|this)\b"
    r"|\b(?:can|could) you back (?:that|this|it) up\b|\bsays who\b"
    r"|\bwho (?:said|says) (?:that|this|so)\b|\bany (?:reference|source|link)s?\b"
    r"|\bwhich (?:organi[sz]ation|study|website|site|source) (?:says|said)\b"
    r"|\bis (?:that|this) from (?:an? |the )?(?:actual |real )?(?:study|source|research)\b",
    re.IGNORECASE,
)

_EVIDENCE = re.compile(
    r"\bis there (?:any |good |real )?(?:evidence|research|proof|science|studies)\b"
    r"|\bwhat does (?:the )?(?:research|evidence|science|data) say\b"
    r"|\b(?:research|studies|evidence|science) (?:on|about|behind|for)\b"
    r"|\b(?:is|isn't) (?:it|that|this) (?:true|a myth|backed|proven|evidence[- ]based)\b"
    r"|\bor is (?:that|it) a myth\b|\bmyth\b"
    r"|\b(?:actually|really) (?:work|works|help|helps|good|bad|real|true|do anything)\b"
    r"|\bany (?:real |good |actual )?(?:evidence|research|studies|proof)\b"
    r"|\b(?:is|isn't) (?:that|it|this) (?:legit|real|overblown|hype)\b"
    r"|\b(?:all |just )?hype\b|\boverblown\b|\bjust something people say\b"
    r"|\bevidence[- ]based\b",
    re.IGNORECASE,
)

_CURRENT = re.compile(
    r"\b(?:latest|newest|current|up[- ]to[- ]date|recent|this year'?s?|new) "
    r"(?:\w+ )?(?:guidelines?|guidance|advice|research|recommendations?|studies|evidence)\b"
    r"|\b(?:anything|something) new (?:come out |out )?(?:lately |recently )?(?:about|on|in)\b"
    r"|\bcome out (?:lately|recently)\b",
    re.IGNORECASE,
)

# Information-request forms. Each must also name a covered topic.
_INFO = re.compile(
    r"\bwhat(?:'s| is| are| does| do)\b"
    r"|\bwhat (?:actually |really |exactly )?happens\b"
    r"|\bhow (?:does|do|can|could|should|to|much|many|long|often)\b"
    r"|\b(?:can|could) you (?:please )?(?:explain|tell me|describe)\b"
    r"|^\W*explain\b|\bexplain (?:what|how)\b"
    r"|\btell me (?:about|more about)\b"
    r"|\bis it (?:true|good|bad|ok|okay|healthy|safe|normal) (?:that|to|for)\b"
    r"|\b(?:are|is) there (?:any |some )?(?:[\w-]+ )?(?:techniques|ways|tips|exercises|"
    r"strategies|methods)\b"
    r"|\b(?:any|some|good|best|got any) (?:tips|advice|techniques|ways|strategies|ideas|"
    r"exercises|suggestions)\b"
    r"|\b(?:tips|advice|techniques|ways|strategies) (?:for|on|to|about)\b"
    r"|\b(?:signs|symptoms|causes|effects|benefits|types) of\b"
    r"|\bdifference between\b"
    r"|\b(?:does|do|can|will) (?:[\w'-]+ ){1,4}(?:really |actually )?(?:help|work|affect|mess|"
    r"ruin|hurt|improve|matter|do anything)\b"
    r"|\bis (?:[\w'-]+ ){1,3}(?:really |actually )?(?:bad|good|healthy|harmful) for\b",
    re.IGNORECASE,
)


# Common texting spellings, so "wat r the stages of grief" reads as a question.
_TEXTING = (
    (re.compile(r"\bwat\b", re.I), "what"),
    (re.compile(r"\bwhats\b", re.I), "what's"),
    (re.compile(r"(?<=\bwhat )r\b|(?<=\bhow )r\b|\br(?= there\b)", re.I), "are"),
    (re.compile(r"\bu\b", re.I), "you"),
    (re.compile(r"\bur\b", re.I), "your"),
    (re.compile(r"\babt\b", re.I), "about"),
)


def _clean(message: str) -> str:
    # Phones type curly apostrophes.
    text = message.replace("\u2019", "'").replace("\u2018", "'").strip()
    for pattern, replacement in _TEXTING:
        text = pattern.sub(replacement, text)
    return text


def route(message: str, *, previous_user_message: str | None = None) -> Route:
    """The route for the person's latest message. `previous_user_message` is
    the query for a source follow-up."""
    text = _clean(message)
    _, advice_allowed = explicit_request(text)
    if not advice_allowed:
        return Route("reflection", False, "", "no_advice")
    if _SOURCE_FOLLOWUP.search(text):
        if previous_user_message and previous_user_message.strip():
            return Route("source_followup", True, _clean(previous_user_message), "source_followup")
        return Route("source_followup", False, "", "nothing_to_source")
    if _REFLECTIVE.search(text):
        return Route("reflection", False, "", "rhetorical_or_personal")
    if not _TOPIC.search(text):
        return Route("casual", False, "", "no_covered_topic")
    if _CURRENT.search(text):
        return Route("current_info", True, text, "current_information")
    if _EVIDENCE.search(text):
        return Route("evidence", True, text, "evidence_request")
    if _INFO.search(text):
        return Route("factual", True, text, "information_request")
    return Route("reflection", False, "", "no_request")
