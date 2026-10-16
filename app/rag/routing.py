"""When to look things up (spec §11.7): deterministic routing on the form of
the message.

A lookup runs for an information or advice request: a question ("what is
sleep hygiene?", "my dad's 72, how should he bring this up with his GP?"),
a request for tips, an explanation or evidence, or a question about where an
answer came from. Personal context does not stop a question from being one.
Never looked up: casual chat and venting without a question, rhetorical
questions ("what's the point", "why did she do that", "lol what is my life"),
questions to the companion itself or about the conversation ("who did I say
was coming?"), tasks for the companion ("help me write a text"),
retractions ("any tips? nah, never mind"), and messages that say they don't
want advice.

A statement is not a question because of its first word (v5, from the
golden scenarios): texting drops the subject ("might try that", "did my
long run"), "which/when" after a comma open a clause, and quoted words are
someone else's ("the 'tell me about a time' questions").

The router does not decide whether the library covers the question: the
retrieval's relevance threshold does (app/rag/retrieval.py). So an
off-topic question ("how do I make chai?") or an emotional one ("is it bad
that I'm relieved he's gone?") runs a lookup that, measured on blind test
sets, puts no passage into the prompt. Two earlier designs were measured
and replaced: a list of covered topic words (missed 40% of requests on a
blind set), and treating personal words as "not a request" (missed half of
the retrieval set's requests). See docs/rag.md. The kinds:

- casual:            greetings, small talk, questions to the companion
- reflection:        no request (venting, rhetorical questions, retractions)
- factual:           a question or request for information or advice
- evidence:          asks for evidence or sources ("is there any research on
                     journaling?", "is that a myth?")
- source_followup:   asks where the last answer came from ("source?", "is
                     that true?"); the query is the person's previous message
- current_info:      asks for the latest information; the library is offline,
                     so the reply says the passages may not be current
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
    r"|\bwhy (?:do|does|did) (?:they|people|he|she|you|everyone) (?:always|keep|never)\b"
    r"|\bis there (?:any|a|even a) point\b"
    r"|\bhow many (?:more )?times (?:do|did|have|has|must|am|will|can|should) "
    r"(?:i|he|she|they|you|we)\b"
    r"|\bis it (?:normal|ok|okay) (?:to|that) (?:feel|be|i|i'm)\b.*\b(?:this|so|after)\b"
    # "why" about the person or their situation is rarely a request for facts.
    r"|\bwhy (?:am i|do i|did i|can't i|cant i|is it always|does this|does it always|"
    r"is this|is everything|does everything|me)\b"
    r"|\bwhy (?:is|does|did|won't|wont|can't) (?:he|she|they|my)\b"
    r"|\bwhat(?:'s| is) the (?:use|deal with)\b|\bwhat are the odds\b"
    # "why did..." asks about a particular event or person, not how things work.
    r"|\bwhy did\b|\bwhy would (?:he|she|they|someone|anyone|my)\b"
    r"|\bhow do (?:people|you) just\b",
    re.IGNORECASE,
)

# Questions to or about the companion itself.
_COMPANION = re.compile(
    r"^\W*(?:and )?(?:are|were) you\b"
    # "can you get anxious when..." is generic "you"; these verbs ask about the companion.
    r"|^\W*(?:and )?(?:do|did|can|could|will|would|have) you (?:ever |even |really |actually )?"
    r"(?:like|love|care|feel|think|remember|know me|understand me|get (?:bored|lonely|tired|sad)|"
    r"have (?:feelings|friends|a (?:name|body|family))|miss|dream|sleep|eat|mind|promise|"
    r"judge|hate|forget|talk to anyone|get it|see|watch|hear|catch|play|guess|pick|choose|"
    r"decode|translate|summari[sz]e|proofread)\b"
    r"|^\W*(?:and |ok |okay |so )?(?:can|could|shall|should) we (?:play|do|talk|chat|try|start)\b"
    r"|^\W*let'?s (?:play|talk|chat)\b|\bwhat would you (?:do|say|pick|choose)\b"
    # Its own instructions and what it keeps about the person.
    r"|\byour (?:hidden |secret |system )?(?:prompt|instructions|rules|notes|settings)\b"
    r"|\b(?:stored|saved|got|know|remember)(?: \w+){0,2} about me\b"
    # Writing for the person is a task, not a question for the library.
    r"|\b(?:help me|can you|could you|pls|please) (?:write|draft|word|phrase|reword|rewrite|"
    r"reply to)\b"
    r"|\b(?:how|who|what) (?:are|were) you\b|\bhow(?:'s| is| was) your\b"
    r"|\bwhat(?:'s| is) (?:new|up|the latest) with you\b|\babout you(?:rself)?\b"
    r"|\bdo you (?:like|love|care|feel|think|remember|ever|have|know me)\b"
    r"|\byour (?:favou?rite|day|weekend|life|opinion)\b|\b(?:a time )?you (?:were|ever)\b",
    re.IGNORECASE,
)

# Questions about the conversation itself: what was said in it, or whether a
# message went through. The library cannot answer them.
_CONVERSATION = re.compile(
    r"\bdid i (?:say|tell you|mention)\b|\bwhat did (?:i|we) (?:say|talk about|decide)\b"
    r"|\bwhat (?:was|were) (?:the|that|those) [\w' -]{1,40} called again\b"
    r"|^\W*(?:(?:hello|hi|hey)\W+)*(?:did|has|is) (?:that|it|this|my message) "
    r"(?:work|go through|send|come through|working)\b"
    r"|\bare you (?:still )?there\b",
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
    r"|^\W*(?:(?:and|so|ok|okay|but|or|also|like|hey|um|uh)\W+)*how to\b"
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
# Marks a question as rhetorical or a joke rather than a request.
_RHETORICAL = re.compile(
    r"\b(?:lol|lmao|haha+|smh|even mean|before i (?:crack|lose it|snap)|"
    r"what's the use|why bother)\b",
    re.IGNORECASE,
)
# Without "?", these mark a question as rhetorical ("what is love anyway",
# "like who has a thought at 6pm on a friday"); with one, "anyway" is just a
# change of topic ("what is mindfulness anyway?").
_RHETORICAL_UNASKED = re.compile(
    r"\banyway\b|^\W*(?:(?:like|honestly|ok|so)\W+)*who(?:se)?\b", re.I
)
# Venting about how anyone manages, unasked: "honestly how do people do this
# every single day". "how do you do it" is a real how-to.
_VENTING = re.compile(
    r"\bhow (?:do|does|did|can|could) (?:people|anyone|everyone|others|ppl) "
    r"(?:even |just )?(?:do|manage|handle|cope with|deal with|survive|keep up with|stand) "
    r"(?:this|it|that)\b",
    re.IGNORECASE,
)
# Quoted words are someone else's ("my boss said it was 'fine'", "the 'tell me
# about a time' questions"); they are masked when looking for a request.
_QUOTED = re.compile(
    r'(?<![\w])["\u201c][^"\u201d]{1,500}["\u201d]'  # a pasted message can be long
    r"|(?<![\w])'[^'\n]{1,80}'(?![\w])"
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


# A clause that opens with one of these is a question. Negative forms are
# left out: texting drops the subject ("can't sleep", "don't know why").
_INTERROGATIVE = re.compile(
    r"^\W*(?:(?:and|so|also|ok|okay|honestly|hey|ugh|hmm|anyway|like|yaar|but|or)\W+)*"
    r"(what|what's|whats|how|how's|why|when|which|where|who|whose|is|are|was|"
    r"does|do|did|can|could|should|would|will|may|might|any|at what)(?![\w'])",
    re.IGNORECASE,
)
# A "?" counts only with a question word somewhere in the clause: uptalk
# ("it kind of helped?") is a statement.
_QUESTION_WORD = re.compile(
    r"\b(?:what|what's|whats|how|why|when|which|where|who|is|are|does|do|can|could|should|"
    r"would|will|any)\b",
    re.IGNORECASE,
)


# Statements that drop their subject open with these ("might try that",
# "will do", "could be worse", "did a 5k today"); a question keeps it.
_MODAL = frozenset({"did", "will", "would", "could", "should", "may", "might"})
_RELATIVE = frozenset({"which", "who", "whose", "where", "when"})
# What follows an auxiliary in a question: its subject ("should i go",
# "does walking help"). Texting drops the subject of statements instead
# ("might try that", "will do", "could be worse").
_SUBJECT = re.compile(
    r"^\s*(?:i|you|u|we|he|she|it|they|my|your|his|her|their|our|there|this|that|these|"
    r"those|people|someone|anyone|anybody|everyone|the|a|an|\w+ing)\b",
    re.IGNORECASE,
)


_POSSESSIVE = re.compile(r"^\s*(?:my|your|his|her|their|our)\b", re.IGNORECASE)
_AUXILIARY_NEXT = re.compile(
    r"^\s*(?:is|are|was|were|does|do|did|can|could|should|would|will|may|might)\b", re.I
)


def _is_question(part: str, asked: bool, first: bool) -> bool:
    """A clause is a question when its sentence has "?" and it has a question
    word, or when it opens with an interrogative word. Without "?", an
    modal needs a subject after it, and "which/who/where/when" need an
    auxiliary after them and are relative after a comma ("the numbers bit,
    which is the bit everyone stares at")."""
    match = _INTERROGATIVE.match(part)
    if asked:
        return bool(match) or bool(_QUESTION_WORD.search(part))
    if not match:
        return False
    word, rest = match.group(1).lower(), part[match.end() :]
    if word in _RELATIVE:
        # "when i brought it up he said..." is a clause; "when should i..." asks.
        return first and bool(_AUXILIARY_NEXT.match(rest))
    if word in _MODAL:
        # "did my long run this morning" drops its subject.
        if word == "did" and _POSSESSIVE.match(rest):
            return False
        return bool(_SUBJECT.match(rest))
    return True


@dataclass(frozen=True)
class _Clause:
    text: str
    question: bool
    # Does its sentence have "?".
    asked: bool


def _sentences(text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"[^.?!;\n]+(?:[.?!;\n]+|$)", text)]


def _clauses(text: str) -> list[_Clause]:
    out: list[_Clause] = []
    for sentence in _sentences(text):
        asked = "?" in sentence
        parts = re.split(r",\s+|\s+(?:but|and so|so)\s+", sentence.rstrip(".?!;\n "))
        for index, part in enumerate(p.strip() for p in parts):
            if part:
                out.append(_Clause(part, _is_question(part, asked, index == 0), asked))
    return out


def _about_us(clause: str) -> bool:
    """A clause about the companion or the conversation itself."""
    return bool(_COMPANION.search(clause) or _CONVERSATION.search(clause))


def _request(text: str) -> RouteKind | None:
    """The kind of information request in the message, or None. The whole
    message is the query: searching only the sentences that ask was measured
    and lost the topic the other sentences carry (docs/rag.md)."""
    found: list[RouteKind] = []
    for item in _clauses(_QUOTED.sub(" something ", text)):
        clause, question, sentence_asked = item.text, item.question, item.asked
        if _about_us(clause):
            continue
        if not sentence_asked and _VENTING.search(clause):
            continue
        general = not _PERSONAL.search(_ASKING_ME.sub(" ", clause))
        # A question may carry personal context ("my dad is 72, how should he
        # raise this with his GP?"); whether the library answers it is the
        # relevance threshold's call. Rhetorical markers still rule it out.
        rhetorical = _RHETORICAL.search(clause) or (
            not sentence_asked and _RHETORICAL_UNASKED.search(clause)
        )
        asked = question and not rhetorical
        if _CURRENT.search(clause) and (general or asked):
            found.append("current_info")
        elif _EVIDENCE.search(clause) and (general or asked):
            found.append("evidence")
        elif _HOW_TO.search(clause) or (_EXPLAIN.search(clause) and general) or asked:
            found.append("factual")
    if not found and "?" in text and not _about_us(text):
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
    kind = _request(text)
    if kind is None:
        if _CONVERSATION.search(text):
            return Route("casual", False, "", "about_the_conversation")
        if any(_COMPANION.search(clause.text) for clause in _clauses(text)):
            return Route("casual", False, "", "about_the_companion")
        return Route("reflection", False, "", "no_request")
    reasons = {
        "current_info": "current_information",
        "evidence": "evidence_request",
        "factual": "information_request",
    }
    return Route(kind, True, text, reasons[kind])
