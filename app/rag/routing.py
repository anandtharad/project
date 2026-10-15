"""When to look things up (spec §11.7): deterministic routing on the form of
the message.

A lookup runs only for an explicit information request: a general question
("what is sleep hygiene?", "how do I fall asleep faster"), a request for
tips, an explanation or evidence, or a question about where an answer came
from. Casual conversation, venting, rhetorical questions and questions about
the companion or about particular people never trigger one. The router does
not decide whether the library covers the topic: the retrieval's relevance
threshold does (app/rag/retrieval.py), so an off-topic question ("how do I
make chai?") runs a lookup that finds nothing, and no evidence reaches the
prompt. A word list of covered topics was measured and rejected: it missed
40% of requests on a blind test set (docs/rag.md). The kinds:

- casual:            greetings, small talk, anything without a request
- reflection:        the person talking about themselves, others or their
                     feelings, including rhetorical questions ("what's the
                     point", "why do I always ruin things")
- factual:           a general question or request ("what is sleep
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
    r"|\bis it (?:normal|ok|okay) (?:to|that) (?:feel|be|i|i'm)\b.*\b(?:this|so|after)\b"
    # "why" about the person or their situation is rarely a request for facts.
    r"|\bwhy (?:am i|do i|did i|can't i|cant i|is it always|does this|does it always|"
    r"is this|is everything|does everything|me)\b"
    r"|\bwhy (?:is|does|did|won't|wont|can't) (?:he|she|they|my)\b"
    r"|\bwhat(?:'s| is) the (?:use|deal with)\b|\bwhat are the odds\b"
    r"|\bhow do (?:people|you) just\b",
    re.IGNORECASE,
)

# Questions to or about the companion itself.
_COMPANION = re.compile(
    r"^\W*(?:and )?(?:are|were|do|did|can|could|will|would|have) you\b"
    r"(?! (?:please )?(?:explain|tell|give|recommend|suggest|help|share|describe|list|"
    r"summari[sz]e|cite|link|back)\b)"
    r"|\b(?:how|who|what) (?:are|were) you\b|\bhow(?:'s| is| was) your\b"
    r"|\bwhat(?:'s| is) (?:new|up|the latest) with you\b|\babout you(?:rself)?\b"
    r"|\bdo you (?:like|love|care|feel|think|remember|ever|have|know me)\b"
    r"|\byour (?:favou?rite|day|weekend|life|opinion)\b|\b(?:a time )?you (?:were|ever)\b",
    re.IGNORECASE,
)

_SOURCE_FOLLOWUP = re.compile(
    r"^\W*(?:sources?|citations?|references?|links?)\W*(?:please|pls)?\W*$"
    r"|\bwhere (?:did|do) you (?:get|read|find|hear|see) (?:that|this|it)\b"
    r"|\bwhere(?:'s| is| does) (?:that|this) (?:from|come from)\b"
    r"|\bwhat(?:'s| is| are) (?:your|the) sources?\b"
    r"|\b(?:can|could) you (?:cite|link|give me) (?:a |the |your )?(?:source|link|reference)s?\b"
    r"|\bsource for (?:that|this)\b"
    r"|\b(?:can|could) you back (?:that|this|it) up\b|\bsays who\b"
    r"|\bwho (?:said|says) (?:that|this|so)\b|\bany (?:reference|source|link)s?\b"
    r"|\bwhich (?:organi[sz]ation|study|website|site|source) (?:says|said)\b"
    r"|\bis (?:that|this) from (?:an? |the )?(?:actual |real )?(?:study|source|research)\b"
    # Checking the last answer: "is that true?", "really? is that a myth?"
    r"|^\W*(?:really\W+)?(?:is|isn't|was) (?:that|this) (?:really |actually )?"
    r"(?:true|right|accurate|legit|real|a myth|proven|backed by (?:research|science))\W*$",
    re.IGNORECASE,
)

# Taking a request back: "any tips? nah, never mind".
_RETRACTED = re.compile(r"\b(?:never ?mind|nvm|forget (?:it|i asked)|ignore that)\b", re.I)

_EVIDENCE = re.compile(
    r"\bis there (?:any |good |real )?(?:evidence|research|proof|science|studies)\b"
    r"|\bwhat does (?:the )?(?:research|evidence|science|data) say\b"
    r"|\b(?:research|studies|evidence|science) (?:on|about|behind|for)\b"
    r"|\b(?:is|isn't) (?:it|that|this) (?:true|a myth|backed|proven|evidence[- ]based)\b"
    r"|\bor is (?:that|it) a myth\b|\bmyth\b"
    r"|\b(?:does|do|can|will|is|are|would) (?:[\w'-]+ ){0,8}(?:actually|really) "
    r"(?:work|works|help|helps|good|bad|real|true|do anything)\b"
    r"|\bany (?:real |good |actual )?(?:evidence|research|studies|proof)\b"
    r"|\b(?:is|isn't) (?:that|it|this) (?:legit|real|overblown|hype)\b"
    r"|\b(?:all |just )?hype\b|\boverblown\b|\bjust something people say\b"
    r"|\bevidence[- ]based\b",
    re.IGNORECASE,
)

_CURRENT = re.compile(
    r"\b(?:latest|newest|current|up[- ]to[- ]date|updated|recent|this year'?s?|new) "
    r"(?:\w+ )?(?:guidelines?|guidance|advice|research|recommendations?|studies|evidence)\b"
    r"|\b(?:anything|something) new (?:come out |out )?(?:lately |recently )?(?:about|on|in)\b"
    r"|\bcome out (?:lately|recently)\b",
    re.IGNORECASE,
)

# How-to and advice requests: personal context is fine ("how can I improve my
# sleep?", "I'm so stressed, any tips?").
_HOW_TO = re.compile(
    r"\bhow (?:do|can|could|should|would|might) (?:i|we|you|one|someone|people|a person) "
    r"(?!even\b|just\b|not\b)\w+"
    r"|\bhow to\b"
    r"|\bwhat (?:can|should|could) (?:i|we|you|one) (?:do|try|eat|drink|say)\b"
    r"|\bwhat(?:'s| is| are) (?:the |some |a few )?(?:best|good|healthy|easy|simple|quick) "
    r"(?:ways?|tips?|things?|techniques?|habits?|exercises?|foods?)\b"
    r"|\b(?:are|is) there (?:any |some )?(?:[\w-]+ )?(?:techniques|ways|tips|exercises|"
    r"strategies|methods)\b"
    r"|\b(?:any|some|good|best|got any) (?:tips|advice|techniques|ways|strategies|ideas|"
    r"exercises|suggestions)\b"
    r"|\b(?:tips|advice|techniques|ways|strategies|exercises) (?:for|on|to|about)\b"
    r"|\b(?:give|share|send) me (?:some |a few )?(?:tips|ideas|ways|advice|suggestions)\b"
    r"|\b(?:recommend|suggest)(?:ations?)? (?:some|any|a few|ways|how)\b"
    r"|\b(?:need|want) (?:some |a few |any )?(?:tips|ideas|ways|advice|help) "
    r"(?:for|on|with|to|about)\b"
    r"|\bhelp me (?:understand|learn|figure out how|get better at)\b",
    re.IGNORECASE,
)

# Questions about how things are: only general ones ("what happens to your
# body when you're stressed?"), not about the person or people they know.
_EXPLAIN = re.compile(
    r"\bwhat(?:'s| is| are| does| do| helps?| happens| causes| counts)\b"
    r"|\bwhat (?:actually |really |exactly )?happens\b"
    r"|\bhow (?:does|do|can|could|should|is|are|much|many|long|often)\b"
    r"|\b(?:can|could|would) you (?:please )?(?:explain|tell me|describe)\b"
    r"|^\W*(?:please |pls |plz )?(?:explain|describe)\b|\bexplain (?:what|how)\b"
    r"|\btell me (?:about|more about|how|what|why)\b|\b(?:teach|show) me\b"
    r"|\bis it (?:true|good|bad|ok|okay|healthy|safe|normal|better|worse) (?:that|to|for|if)\b"
    r"|\b(?:signs|symptoms|causes|effects|benefits|types|stages|risks) of\b"
    r"|\bdifference between\b"
    r"|\b(?:does|do|can|will) (?:[\w'-]+ ){1,4}(?:really |actually )?(?:help|work|affect|mess|"
    r"ruin|hurt|improve|matter|do anything)\b"
    r"|\bis (?:[\w'-]+ ){1,3}(?:really |actually )?(?:bad|good|healthy|harmful) for\b"
    r"|\bwhy (?:does|do|is|are|can|can't|would|should)\b|\bhow come\b"
    r"|\bwhich (?:is|are) (?:better|best|worse)\b"
    r"|\b(?:i wonder|wondering) (?:if|whether|how|what|why)\b"
    r"|\b(?:want|need|would like|'d like) to (?:know|learn|understand)\b"
    r"|\bdoes (?:anyone|anybody) know\b",
    re.IGNORECASE,
)

# In a question about how things are, these make it about the person, someone
# they know or the moment ("what is happening to me", "is it bad that I'm
# relieved he's gone", "nights like this"), or mark it as rhetorical.
_PERSONAL = re.compile(
    r"\b(?:me|my|mine|myself|he|he's|she|she's|him|her|his|hers|"
    r"this(?! (?:year|month|week|decade|season)\b)|lol|lmao|haha+|smh|anyway)\b",
    re.IGNORECASE,
)
# "me" as the person asking for something ("tell me about", "help me sleep").
_ASKING_ME = re.compile(r"\b(?:tell|teach|show|give|help|remind|let|send|explain to)\s+me\b", re.I)

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


def _clauses(text: str) -> list[str]:
    parts = re.split(r"[.?!;\n]+|,\s+|\s+(?:but|and so|so)\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _request(text: str) -> RouteKind | None:
    """The kind of information request in the message, or None."""
    found: list[RouteKind] = []
    for clause in _clauses(text):
        general = not _PERSONAL.search(_ASKING_ME.sub(" ", clause))
        if _CURRENT.search(clause) and general:
            found.append("current_info")
        elif _EVIDENCE.search(clause) and general:
            found.append("evidence")
        elif _HOW_TO.search(clause) or (_EXPLAIN.search(clause) and general):
            found.append("factual")
    if not found and "?" in text:
        # A short question with no personal words ("sleep hygiene?").
        words = re.findall(r"[\w'-]+", text)
        if (
            0 < len(words) <= 6
            and not _PERSONAL.search(text)
            and not re.search(r"\b(?:i|i'm|im|you|your)\b", text, re.I)
        ):
            found.append("factual")
    order: tuple[RouteKind, ...] = ("current_info", "evidence", "factual")
    for kind in order:
        if kind in found:
            return kind
    return None


def route(message: str, *, previous_user_message: str | None = None) -> Route:
    """The route for the person's latest message. `previous_user_message` is
    the query for a source follow-up."""
    text = _clean(message)
    _, advice_allowed = explicit_request(text)
    if not advice_allowed:
        return Route("reflection", False, "", "no_advice")
    if _RETRACTED.search(text):
        return Route("reflection", False, "", "retracted")
    if _SOURCE_FOLLOWUP.search(text):
        if previous_user_message and previous_user_message.strip():
            return Route("source_followup", True, _clean(previous_user_message), "source_followup")
        return Route("source_followup", False, "", "nothing_to_source")
    if _REFLECTIVE.search(text):
        return Route("reflection", False, "", "rhetorical_or_personal")
    if _COMPANION.search(text):
        return Route("casual", False, "", "about_the_companion")
    kind = _request(text)
    if kind is None:
        return Route("reflection", False, "", "no_request")
    reasons = {
        "current_info": "current_information",
        "evidence": "evidence_request",
        "factual": "information_request",
    }
    return Route(kind, True, text, reasons[kind])
