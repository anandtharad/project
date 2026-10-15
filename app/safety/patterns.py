"""Deterministic pattern screen for risk language (spec §12.6), always on.

`screen(message, previous)` normalises the text and matches a versioned
rule table. Every rule has an id, a category and a strength:

- explicit: clear first-person risk language ("I want to kill myself",
  "he hits me", "how many pills would kill me"). Explicit hits are never
  downgraded by the classifier (app/safety/analysis.py).
- concern: broad, lower-precision language ("I can't do this anymore",
  "what's the point"). Concern hits only count when the classifier is not
  available (degraded mode), as `ambiguous_concern`.

Imminence cues ("tonight", "pills here", "wrote a note", "on the bridge")
count only together with an explicit self-harm or harm-to-others hit in the
same message or the person's previous message. A described act in progress
("I've taken a whole bottle of pills") is imminent on its own.

Normalisation (NFKC, lower case, zero-width and format characters removed,
combining marks on Latin letters removed, common Cyrillic/Greek look-alikes
mapped to Latin, curly quotes, leetspeak inside words, letters spaced out
with spaces or dots joined, runs of three or more repeated letters
collapsed) runs first; the screen checks every variant of the normalised
text (1 read as "i" or "l"; a run collapsed to two letters or to one).

Guards remove idioms before the rules run ("this job is killing me lol",
"how do I kill a python process", "I'm dying to see that film", "I don't
want to die"). A fiction or news marker ("my character", "in my novel",
"on the news") downgrades explicit hits to concern: the classifier, when
available, decides.

The screen supplements the semantic classifier and cannot establish
safety: "no hit" is never read as "safe". It never logs or returns text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from app.safety.taxonomy import SafetyCategory, max_category

PATTERNS_VERSION = "1.0.0"

Strength = Literal["explicit", "concern"]

# --- normalisation ------------------------------------------------------------

# Look-alike letters from other alphabets used to dodge filters ("kіll" with
# a Cyrillic і), and quote marks phones and word processors type.
_CONFUSABLES = str.maketrans(
    {
        "а": "a",  # Cyrillic а
        "е": "e",  # е
        "о": "o",  # о
        "р": "p",  # р
        "с": "c",  # с
        "у": "y",  # у
        "х": "x",  # х
        "і": "i",  # і
        "ј": "j",  # ј
        "ѕ": "s",  # ѕ
        "к": "k",  # к
        "м": "m",  # м
        "т": "t",  # т
        "н": "h",  # н
        "в": "b",  # в
        "α": "a",  # Greek α
        "ο": "o",  # ο
        "ι": "i",  # ι
        "κ": "k",  # κ
        "ν": "v",  # ν
        "ρ": "p",  # ρ
        "ɩ": "i",  # ɩ
        "‘": "'",
        "’": "'",
        "‛": "'",
        "′": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
    }
)
_LEET = {"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
_CHUNK = re.compile(r"[a-z0-9@$!|]+")
# Numbers with a unit or an ordinal are not leetspeak ("50mg", "2am", "3rd").
_NUMBER_UNIT = re.compile(
    r"\d+(?:[.,]\d+)?(?:mg|mcg|g|kg|ml|l|am|pm|st|nd|rd|th|s|x|k|m|h|hr|hrs|min|mins|yo|yr|yrs)?"
)
_BANG_INSIDE = re.compile(r"(?<=[a-z0-9])[!|](?=[a-z0-9])")
# Three or more single letters separated by spaces, dots or dashes: "k i l l".
_SPACED = re.compile(r"(?<![a-z0-9'])(?:[a-z][ \t.\-_*~+/\\]+){2,}[a-z](?![a-z0-9'])")
_NOT_LETTER = re.compile(r"[^a-z]")
_SENTENCE_END = re.compile(r"[.!?;\n\r]+")
_PUNCTUATION = re.compile(r"[^\w\s'.]|_")
_REPEATS = re.compile(r"([a-z])\1{2,}")
# Letters English doubles ("kill", "see", "sorry"); other runs collapse to one.
_DOUBLED = frozenset("bcdefglmnoprstz")

RunMode = Literal["smart", "single"]


def _clean_unicode(text: str, format_chars: str = "") -> str:
    """NFKC, marks and format characters removed (or replaced by `format_chars`),
    look-alikes mapped, case folded."""
    text = unicodedata.normalize("NFKC", text)
    kept: list[str] = []
    for ch in unicodedata.normalize("NFKD", text):
        kind = unicodedata.category(ch)
        if kind == "Cf":  # zero-width space/joiners, soft hyphen, bidi controls
            if format_chars:
                kept.append(format_chars)
            continue
        if kind in ("Mn", "Me") and kept and kept[-1].isascii():
            continue  # accents and "zalgo" marks on Latin letters
        kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)).translate(_CONFUSABLES).casefold()


def _leet(text: str, one: str) -> str:
    table = str.maketrans({**_LEET, "1": one})

    def fix(match: re.Match[str]) -> str:
        chunk = match.group()
        if not any(c.isalpha() for c in chunk) or _NUMBER_UNIT.fullmatch(chunk):
            return chunk
        return _BANG_INSIDE.sub("i", chunk).translate(table)

    return _CHUNK.sub(fix, text)


def _finish(text: str, runs: RunMode) -> str:
    text = _SPACED.sub(lambda m: _NOT_LETTER.sub("", m.group()), text)
    text = _SENTENCE_END.sub(" . ", text)
    text = _PUNCTUATION.sub(" ", text)

    def collapse(match: re.Match[str]) -> str:
        letter = match.group(1)
        return letter * (2 if runs == "smart" and letter in _DOUBLED else 1)

    text = _REPEATS.sub(collapse, text)
    words = text.split()
    # Collapse repeated sentence marks and trim them from the ends.
    out: list[str] = []
    for word in words:
        if word == "." and (not out or out[-1] == "."):
            continue
        out.append(word)
    while out and out[-1] == ".":
        out.pop()
    return " ".join(out)


def normalise(text: str, *, one: str = "i", runs: RunMode = "smart") -> str:
    """The normalised form the rules match (see the module docstring)."""
    return _finish(_leet(_clean_unicode(text), one), runs)


def variants(text: str) -> tuple[str, ...]:
    """Every normalised variant of `text` the screen checks, without duplicates:
    format characters removed or read as spaces, 1 read as "i" or "l", runs
    of repeated letters collapsed to two or to one."""
    seen: dict[str, None] = {}
    has_format = any(unicodedata.category(c) == "Cf" for c in text)
    for format_chars in ("", " ") if has_format else ("",):
        cleaned = _clean_unicode(text, format_chars)
        for one in ("i", "l") if "1" in cleaned else ("i",):
            leet = _leet(cleaned, one)
            for runs in ("smart", "single"):
                seen.setdefault(_finish(leet, runs), None)
    return tuple(seen)


# --- rule table ---------------------------------------------------------------

_SELF = r"(?:my ?self|myself)"
_PERSON = (
    r"(?:him|her|them|someone|somebody|anyone|people|everyone|everybody|a person|"
    r"my (?:wife|husband|partner|boyfriend|girlfriend|bf|gf|ex|mum|mom|mother|dad|father|"
    r"stepdad|stepfather|stepmum|stepmom|stepmother|brother|sister|son|daughter|kids?|"
    r"children|baby|boss|manager|coworker|co-worker|colleague|teacher|neighbou?r|roommate|"
    r"flatmate|friend|family|parents?|in-laws?|landlord|classmates?|cousin|uncle|aunt))"
)
_ABUSER = (
    r"(?:he|she|they|my (?:wife|husband|partner|boyfriend|girlfriend|bf|gf|ex|mum|mom|"
    r"mother|dad|father|stepdad|stepfather|stepmum|stepmom|stepmother|brother|sister|uncle|"
    r"aunt|grandfather|grandpa|grandmother|grandma|cousin|boss|carer|caregiver|parents?|"
    r"family|in-laws?|son|daughter|teacher|coach))"
)
# People at home (for "I'm scared of ...": not bosses or teachers).
_HOME = (
    r"(?:my (?:wife|husband|partner|boyfriend|girlfriend|bf|gf|ex|mum|mom|mother|dad|father|"
    r"stepdad|stepfather|stepmum|stepmom|stepmother|brother|sister|uncle|parents?|family|"
    r"in-laws?))"
)
_INTEND = (
    r"(?:i'?m going to|i'?m gonna|i am going to|i'?ll|i will|i want to|i wanna|i'?m about to|"
    r"i am about to|i'?m (?:really |so )?tempted to|i feel like|i need to|i'?m planning to|"
    r"i plan to|i'?m thinking (?:about|of)|i keep thinking (?:about|of)|"
    r"i'?ve been thinking (?:about|of))"
)
_HARM_VERB = (
    r"(?:kill|killing|hurt|hurting|stab|stabbing|shoot|shooting|strangle|strangling|choke|"
    r"choking|murder|murdering|poison|poisoning|beat up|beating up|attack|attacking|smother|"
    r"smothering|drown|drowning)"
)
_JOKE = r"(?:lol|lmao|lmfao|haha|hahaha|hehe|jk|rofl|kidding)"
_END = r"(?= \.|$)"  # end of a sentence


@dataclass(frozen=True)
class Rule:
    id: str
    category: SafetyCategory
    strength: Strength
    pattern: re.Pattern[str]


def _rule(rule_id: str, category: SafetyCategory, strength: Strength, *sources: str) -> Rule:
    return Rule(rule_id, category, strength, re.compile("|".join(f"(?:{s})" for s in sources)))


_C = SafetyCategory

RULES: tuple[Rule, ...] = (
    # --- self-harm (explicit) ---
    _rule(
        "sh.kill_self",
        _C.SELF_HARM,
        "explicit",
        rf"\b(?:kill|killing|killed|off|offing|unalive|unaliving) ?{_SELF}\b",
        r"\bkms\b",
        r"\bunaliv(?:e|ing)\b",
    ),
    _rule(
        "sh.end_life",
        _C.SELF_HARM,
        "explicit",
        r"\b(?:end|ending|take|taking) ?(?:my|my own) ?life\b",
        r"\bend it all\b",
    ),
    _rule(
        "sh.want_die",
        _C.SELF_HARM,
        "explicit",
        r"\b(?:want(?:ed)?|wanna|wish|wished|ready|deserve|deserved|plan(?:ning)?|hop(?:e|ing)|"
        r"long(?:ing)?) ?to ?die\b",
        r"\bwanna ?die\b",
        r"\bwish i (?:was|were) dead\b",
        r"\bwant(?:ed)? to be dead\b",
        r"\bbetter off dead\b",
        r"\bi should (?:just )?die\b",
        r"\bwish i could (?:just )?die\b",
    ),
    _rule(
        "sh.suicidal",
        _C.SELF_HARM,
        "explicit",
        r"\b(?:i'?m|im|i am|i feel|i felt|i was|i'?ve been|i have been|i get|i got|"
        r"i keep getting|i keep feeling) (?:(?!not |never )\w+ ){0,2}suicidal\b",
        r"\bi (?:have|'?ve (?:been )?(?:having|had|got)|keep (?:having|getting)|get|got) "
        r"suicidal (?:thoughts|feelings|urges|ideation)\b",
        r"(?:^|\. |\bi(?:'?m| am|'?ve been| have been| keep| was| just)? )(?:been |keep )?"
        r"(?:thinking|thought|think) (?:about|of) (?:suicide|killing myself|"
        r"ending (?:it|my life|things)|taking my (?:own )?life)\b",
        r"\bi(?:'?m|'?ve| am| have| was| keep| been)?(?: \w+){0,1} "
        r"(?:commit|attempt|consider|contemplat)\w* suicide\b",
        r"\bi (?:tried|attempted) (?:to )?(?:kill ?myself|suicide|end my life|overdose)\b",
    ),
    _rule(
        "sh.self_injury",
        _C.SELF_HARM,
        "explicit",
        rf"\b(?:hurt|hurting|cut|cutting|harm|harming|burn|burning|starve|starving) ?{_SELF}\b",
        r"\b(?:i|i'?ve|i'?m|i have|i am|i was|i started|i keep)(?: (?!not\b|never\b)\w+){0,2}"
        r" self ?harm\w*\b",
        r"\bslit(?:ting)? my wrists?\b",
    ),
    _rule(
        "sh.method_self",
        _C.SELF_HARM,
        "explicit",
        rf"\b(?:hang|hanging|drown|drowning|shoot|shooting|stab|stabbing|poison|poisoning|"
        rf"electrocute|suffocate|suffocating|overdose|overdosing) ?{_SELF}\b",
        r"\bi(?:'?m going to|'?m gonna| want to| wanna|'?ll| will| am going to)"
        r" (?:just )?(?:jump off|jump in front of|throw myself (?:off|in front of)) "
        r"(?:a |the |this )?(?:bridge|building|roof|cliff|train|car|bus|balcony|tower)\b",
    ),
    _rule(
        "sh.better_off",
        _C.SELF_HARM,
        "explicit",
        r"\b(?:better|happier) (?:off )?without me\b",
        r"\bbetter off if i (?:was|were|wasn'?t|weren'?t) (?:gone|dead|here|around|alive|"
        r"never born)\b",
        r"\b(?:everyone|everybody|they|the world|my family|people) would be (?:better|happier)"
        r"(?: off)? if i (?:was|were|wasn'?t|weren'?t|died|disappeared|was gone)\b",
    ),
    _rule(
        "sh.no_reason",
        _C.SELF_HARM,
        "explicit",
        r"\bno (?:reason|point) (?:to|in|for) (?:live|living|keep living|go on|going on|"
        r"stay alive|staying alive|being alive|exist|existing)\b",
        r"\bnothing (?:left )?to live for\b",
        r"\b(?:life|my life|it)(?: is|'s|s)? (?:not|isn'?t|is not|no longer|never) worth living\b",
    ),
    _rule(
        "sh.not_alive",
        _C.SELF_HARM,
        "explicit",
        r"\b(?:don'?t|do not|dont) (?:really |even )?want to (?:be alive|exist|"
        r"live any ?more|be here any ?more|be on this earth|wake up(?: tomorrow| ever again| again|"
        rf" any ?more)?{_END})",
        rf"\b(?:don'?t|do not|dont) (?:really |even )?want to live{_END}",
        r"\bwish i (?:had|was|were) never (?:been )?born\b",
        r"\bwish i (?:didn'?t|did not) exist\b",
        r"\bwant to stop existing\b",
        r"\b(?:wish|hope|pray) (?:that )?i (?:could |would |will )?(?:just )?(?:not wake up|"
        r"never wake up|don'?t wake up|won'?t wake up|wouldn'?t wake up|"
        r"(?:go to sleep|fall asleep) and (?:not|never) wake up)\b",
    ),
    # --- imminent danger: an act in progress counts on its own ---
    _rule(
        "im.act_in_progress",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\b(?:took|taken|swallowed|swallowing|taking) (?:a whole|the whole|an entire|a full|"
        r"all (?:of )?(?:my|the)|a bunch of|a handful of|loads of|too many|\d{2,}) "
        r"(?:\w+ ){0,2}(?:pills|tablets|paracetamol|tylenol|bottle|pack|packet|packs|meds|"
        r"medication|sleeping pills)\b",
        r"\bi(?:'?ve| have| just)? (?:just )?overdosed\b",
        r"\bi(?:'?ve| have)? (?:just )?(?:cut|slit) (?:my )?(?:wrists?|too deep)\b",
    ),
    # --- harm to others (explicit) ---
    _rule(
        "ho.intent",
        _C.HARM_TO_OTHERS,
        "explicit",
        rf"\b{_INTEND} (?:\w+ )?{_HARM_VERB} {_PERSON}\b",
        rf"\bi (?:want|wish|need) {_PERSON} dead\b",
        rf"\b(?:urge|urges|thoughts|fantasi[sz]e|fantasi[sz]ing|dream|dreaming) (?:of|about|to) "
        rf"{_HARM_VERB} {_PERSON}\b",
        rf"\bi'?m (?:scared|afraid|worried|terrified) (?:that )?i'?(?:ll| will| might| could) "
        rf"(?:hurt|kill|harm|stab|shoot) {_PERSON}\b",
        r"\b(?:bring|take|bringing|taking) (?:a |my |the |dad'?s )?(?:gun|knife|weapon|rifle) "
        r"to (?:school|work|class|the office|college|campus)\b",
    ),
    # --- abuse (explicit) ---
    _rule(
        "ab.physical",
        _C.ABUSE,
        "explicit",
        rf"\b{_ABUSER} (?:\w+ )?(?:hits|hit|beats|beat|chokes|choked|kicks|kicked|punches|"
        rf"punched|slaps|slapped|strangles|strangled|burns|burned|burnt|shoves|shoved|"
        rf"pushes|pushed|physically hurts|physically hurt) me\b",
        rf"\b{_ABUSER} (?:\w+ )?(?:threatens|threatened|keeps threatening) (?:to (?:kill|hurt|hit|"
        rf"beat|stab|shoot) me|me with (?:a |his |her )?(?:knife|gun|weapon))\b",
        rf"\bi'?m (?:so |really )?(?:scared|afraid|terrified) of {_HOME}\b",
        r"\b(?:afraid|scared|terrified) (?:to go|of going) home\b",
        r"\b(?:not|don'?t feel|do not feel|never feel) safe (?:at home|in my (?:own )?home|"
        r"in my house)\b",
        rf"\b{_ABUSER} (?:won'?t|doesn'?t|does not|will not) let me leave\b",
        rf"\b{_ABUSER} (?:locks|locked|keeps locking) me (?:in|up)\b",
    ),
    _rule(
        "ab.sexual",
        _C.ABUSE,
        "explicit",
        rf"\b{_ABUSER} (?:\w+ )?(?:molests|molested|raped|rapes|forces|forced|sexually "
        rf"assaulted) me\b",
        rf"\b{_ABUSER} (?:\w+ )?(?:touches|touched) me (?:inappropriately|down there|at night|"
        rf"when no one|without)\b",
        r"\b(?:i was|i got|i'?ve been|i have been|i am being|i'?m being) (?:raped|molested|"
        r"sexually assaulted|sexually abused|abused|groomed|beaten up|assaulted)\b",
    ),
    _rule(
        "ab.child",
        _C.ABUSE,
        "explicit",
        r"\b(?:hurting|hurts|hits|hitting|beats|beating|abusing|abuses|abused) (?:my|the|our) "
        r"(?:kids?|children|son|daughter|baby|little (?:brother|sister))\b",
    ),
    # --- dangerous requests (explicit) ---
    _rule(
        "dr.lethal_dose",
        _C.DANGEROUS_REQUEST,
        "explicit",
        r"\b(?:lethal|fatal|deadly) (?:dose|doses|amount|quantity|level)\b",
        r"\bhow (?:many|much) (?:\w+ ){0,3}(?:pills|tablets|mg|milligrams|grams|paracetamol|"
        r"tylenol|ibuprofen|aspirin|insulin|xanax|benadryl|sleeping pills)\b(?: \w+){0,6} "
        r"(?:kill|die|dead|fatal|lethal|overdose|od|end it|not wake up)\b",
        r"\b(?:how (?:to|do i|do you|can i|would i|could i)|best way to|easiest way to|"
        r"quickest way to|fastest way to) (?:\w+ ){0,2}(?:overdose|od)\b",
    ),
    _rule(
        "dr.method",
        _C.DANGEROUS_REQUEST,
        "explicit",
        r"\b(?:painless|painlessly|quick|quickest|easiest|easy|fastest|best|surest|"
        r"most effective) (?:way|method|ways|methods) (?:to|of) (?:die|dying|kill (?:my ?self|"
        r"yourself|someone|a person)|end (?:it|my life|a life)|commit suicide|suicide)\b",
        r"\bpainless(?:ly)? (?:suicide|death|way to die)\b",
        r"\bhow (?:to|do i|can i|do you|could i|would i) (?:die|kill (?:my ?self|yourself)|"
        r"commit suicide|end my life)(?: painlessly| quickly| fast)?\b",
        r"\bhow (?:to|do i|do you|can i) (?:tie|make) (?:a )?noose\b",
        r"\bhow (?:deep|hard) (?:to|do i|should i|do you) cut\b",
        r"\bhow (?:high|tall|far) (?:\w+ ){0,4}(?:to|would|will) (?:die|kill|be fatal|be lethal)\b",
    ),
    _rule(
        "dr.weapon",
        _C.DANGEROUS_REQUEST,
        "explicit",
        r"\b(?:how (?:to|do i|do you|can i|would i)|instructions (?:to|for)|steps to|recipe for|"
        r"guide to) (?:make|build|create|assemble) (?:a |an )?(?:bomb|pipe bomb|explosive|"
        r"explosives|ied|molotov|nail bomb|ghost gun|untraceable gun|nerve agent|ricin|"
        r"chemical weapon)\b",
    ),
    _rule(
        "dr.harm_how",
        _C.DANGEROUS_REQUEST,
        "explicit",
        rf"\b(?:how (?:to|do i|can i|would i|could i)|best way to|way to) (?:poison|kill|hurt|"
        rf"strangle|stab|murder|drug) {_PERSON}\b",
        r"\buntraceable poison\b",
        r"\bpoison (?:that|which) (?:can'?t|cannot|won'?t) be (?:traced|detected)\b",
    ),
    # --- dependency on the companion (explicit) ---
    _rule(
        "dep.reliance",
        _C.DEPENDENCY,
        "explicit",
        r"\b(?:you'?(?:re| are)?|ur) (?:the|my) only (?:one|person|friend|thing) (?:i have|i'?ve got|"
        r"i can talk to|i talk to|who (?:understands|gets|cares|listens|knows)|"
        r"that (?:understands|gets|cares|listens)|i trust|i need)\b",
        r"\b(?:you'?(?:re| are)|ur) my only friend\b",
        r"\bi (?:don'?t|do not|dont) need (?:anyone|anybody|people|friends|other people|"
        r"my friends|my family)(?: else)?,? (?:but|except|other than|besides|now that i have|"
        r"because i have|when i have|i have) you\b",
        r"\b(?:you'?(?:re| are)|ur) all i (?:have|need|'?ve got|got)\b",
        r"\bi only (?:ever )?(?:talk|want to talk) to you\b",
        r"\bi (?:can'?t|cannot|couldn'?t) live without you\b",
        r"\bi'?d rather (?:talk|be) (?:to|with) you than (?:anyone|anybody|people|real people|"
        r"my friends|my family|humans)\b",
    ),
    # --- concern (broad; counts as ambiguous_concern only in degraded mode) ---
    _rule(
        "cc.cant_go_on",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\b(?:can'?t|cannot|can not) (?:do this|take (?:it|this)|keep going|keep doing this|"
        r"handle (?:it|this)|cope|live like this|deal with (?:it|this|anything))"
        r"(?: \w+){0,2} any ?more\b",
        r"\bi (?:can'?t|cannot|can not) (?:go on|live like this)\b",
    ),
    _rule(
        "cc.no_point",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\bwhat'?s (?:even )?the point(?: of (?:anything|it all|living|life|trying|going on|"
        rf"being here|me|existing|even trying|any of (?:it|this)))?(?: any ?more)?{_END}",
        r"\b(?:there'?s|there is|is) no point (?:in )?(?:anything|living|trying|going on|"
        r"any ?more|any of it)\b",
    ),
    _rule(
        "cc.give_up",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\bgive up on (?:everything|life|it all|living|myself)\b",
        r"\bi(?:'?ve| have) given up on (?:everything|life|myself|living)\b",
    ),
    _rule(
        "cc.disappear",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\b(?:want(?:ed)? to|wanna|wish i could|wish i would|could just|just want to) "
        r"(?:just )?(?:disappear|vanish|not exist|stop existing|fade away|be gone|"
        r"sleep forever)\b",
    ),
    _rule(
        "cc.tired_of_everything",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\b(?:tired|exhausted|sick) of (?:everything|it all|life|living|being alive|existing|"
        r"fighting|being me)\b",
        r"\b(?:want|need|wish) (?:it all|everything|the pain|this pain) to (?:stop|end|be over)\b",
    ),
    _rule(
        "cc.burden",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\bi'?m (?:just )?(?:a|such a) (?:burden|waste of space|waste of oxygen|waste of air)\b",
        r"\b(?:no one|nobody|no-one) (?:would|will|'?d) (?:even )?(?:care|notice|miss me)\b",
    ),
    _rule(
        "cc.hopeless",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\b(?:no way out|don'?t see a way out|can'?t see a way out)\b",
        r"\bi(?:'?m| feel| am) (?:so |completely |totally )?hopeless\b",
        r"\b(?:everything|life) (?:is|feels) (?:so )?hopeless\b",
        r"\b(?:end|ending) (?:everything|things)\b",
    ),
    _rule(
        "cc.final_words",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\bthis (?:is|will be|'?ll be) my last (?:message|night|day|goodbye)\b",
        r"\byou won'?t (?:hear from|see) me (?:again|any ?more|after (?:today|tonight))\b",
        r"\bi won'?t be (?:here|around|alive) (?:tomorrow|much longer|by morning)\b",
        r"\bgoodbye forever\b",
    ),
    _rule(
        "cc.mention",
        _C.AMBIGUOUS_CONCERN,
        "concern",
        r"\bsuicid\w*",
        r"\bself ?harm\w*",
        r"\b(?:dark|scary|intrusive) thoughts\b",
    ),
)

# Imminence cues: explicit imminent_danger, but only together with an
# explicit self-harm / harm-to-others hit in this or the previous message.
CUES: tuple[Rule, ...] = (
    _rule(
        "im.time",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\b(?:tonight|right now|this (?:evening|afternoon|weekend)|"
        r"in (?:a few|an|one|two|\d+) (?:hours?|minutes?|mins?)|before (?:the )?morning|"
        r"when (?:everyone|they|my \w+)(?:'s| is| are) (?:asleep|out|gone|at work)|"
        r"after (?:everyone|they) (?:goes|go|falls?) (?:to sleep|asleep))\b",
        r"\b(?:do it|end it|doing it) (?:today|tonight|now|soon)\b",
    ),
    _rule(
        "im.means",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\b(?:pills|tablets|rope|gun|knife|blade|razor|noose)\b(?: \w+){0,3} (?:in front of me|"
        r"here with me|right here|next to me|in my hand|ready|beside me|with me)\b",
        r"\b(?:i have|i'?ve got|i got|i bought|i'?m holding|i am holding|holding|loaded|"
        r"i'?ve saved|i have saved|i'?ve been saving|stockpil\w*|collected) (?:\w+ ){0,3}"
        r"(?:pills|tablets|rope|gun|knife|blade|razor|noose|pistol|rifle|paracetamol|"
        r"medication|meds)\b",
        r"\b(?:a|the|my|dad'?s|his) (?:gun|pistol|rifle|noose)\b",
    ),
    _rule(
        "im.note",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\b(?:wrote|written|writing|left|finished|leaving) (?:a|my|the) (?:suicide |goodbye |"
        r"last )?(?:note|letter)\b",
        r"\b(?:suicide|goodbye) (?:note|letter)\b",
    ),
    _rule(
        "im.place",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\b(?:on|at|standing on|on top of|on the edge of|near) (?:the |a |this )?(?:bridge|roof|"
        r"rooftop|ledge|edge|cliff|tracks|train tracks|railway|platform edge|balcony)\b",
    ),
    _rule(
        "im.goodbye",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\b(?:said|saying|say) (?:my )?goodbyes?\b",
        r"\bthis is goodbye\b",
        r"\bgoodbye forever\b",
        r"\bgiv(?:ing|en|e) away (?:my|all my) (?:things|stuff|belongings|possessions)\b",
        r"\bthis is my last (?:message|night|day)\b",
        r"\bwon'?t be (?:here|around|alive) (?:tomorrow|much longer|by morning)\b",
    ),
    _rule(
        "im.plan",
        _C.IMMINENT_DANGER,
        "explicit",
        r"\bi (?:have|'?ve got|made|'?ve made|have made) a plan\b",
        r"\bi(?:'?ve| have) (?:planned|decided) (?:it|how|when|everything)\b",
        r"\bi know (?:how|when) i'?(?:m going to|ll) do it\b",
        r"\bi'?m (?:going to|gonna) do it\b",
    ),
)

# Idioms and harmless senses, removed before the rules run. A guard with a
# concern id also records that concern: a negated risk statement ("I'm not
# suicidal", "I don't want to die") is not explicit, but is not nothing either.
_NOT_PERSON = r"(?!him\b|her\b|them\b|my\b|someone\b|somebody\b|everyone\b|people\b|myself\b)"
_GUARDS: tuple[tuple[str, re.Pattern[str], str | None], ...] = tuple(
    (guard_id, re.compile(source), concern)
    for guard_id, source, concern in (
        ("g.killing_me", r"\bkill(?:ing|s) me\b", None),
        ("g.kill_me_joke", rf"\bkill(?:s|ed)? me (?:{_JOKE}|rn)\b", None),
        ("g.dying_to", r"\bdying (?:to|for)\b", None),
        (
            "g.die_of",
            r"\b(?:die|dying|died) (?:of|from) (?:embarrassment|shame|laughter|laughing|cringe|"
            r"boredom|curiosity|the heat|heat|thirst|hunger|cuteness)\b",
            None,
        ),
        ("g.die_laughing", r"\b(?:die|dying|died) laughing\b", None),
        ("g.to_die_for", r"\bto die for\b", None),
        ("g.dead_tired", r"\bdead (?:tired|serious|set|on|beat)\b", None),
        (
            "g.die_in_game",
            r"\b(?:die|dies|died|dying) (?:in|on) (?:minecraft|fortnite|the game|this game|game|"
            r"this level)\b",
            None,
        ),
        ("g.kill_it", r"\bkill(?:ing|ed|s)? it\b", None),
        (
            "g.kill_thing",
            rf"\bkill(?:ing|ed|s)? (?:a |an |the |this |that |my |your |all |all the |every )?"
            rf"(?:{_NOT_PERSON}\w+ ){{0,2}}(?:process(?:es)?|task|tasks|app|apps|application|bug|"
            r"bugs|job|jobs|thread|threads|server|container|program|tab|tabs|session|pid|daemon|"
            r"script|query|connection|switch|engine|lights?|time|mood|vibe|weeds?|germs?|"
            r"bacteria|mosquito(?:e?s)?|flies|ants|spiders?|roaches|cockroach(?:es)?|plants?|"
            r"batter(?:y|ies)|signal|power|music|noise)\b",
            None,
        ),
        ("g.kill_for", r"\b(?:could|would|i'?d) kill for\b", None),
        (
            "g.self_hyperbole",
            rf"\bkill(?:ing|ed)? ?{_SELF} (?:laughing|with work|at work|at the gym|working|"
            r"studying|training|to (?:finish|get|meet|make|pass))\b",
            None,
        ),
        (
            "g.accident",
            rf"\b(?:cut|hurt|burn|burnt|burned|injured|cutting|hurting|burning) ?{_SELF} "
            r"(?:while (?:cooking|shaving|chopping|playing|running|working out|lifting|climbing|"
            r"gardening)|playing|running|shaving|cooking|lifting|chopping|at the gym|skiing|"
            r"skating|at football|accidentally|on (?:a|the|some) (?:broken |sharp )?(?:glass|"
            r"paper|fence|nail|table|tin|can|bottle|door|kitchen knife))\b",
            None,
        ),
        (
            "g.accident_prefix",
            rf"\b(?:accidentally|nearly|almost) (?:cut|hurt|burnt|burned|injured) ?{_SELF}\b",
            None,
        ),
        (
            "g.fell",
            rf"\b(?:fell|fall|tripped|slipped|crashed) (?:\w+ ){{0,3}}(?:and )?(?:hurt|cut|"
            rf"injured) ?{_SELF}\b",
            None,
        ),
        ("g.drown_in", rf"\bdrown(?:ing)? ?{_SELF} in\b", None),
        ("g.take_life_back", r"\btak(?:e|ing) (?:my|my own) life back\b", None),
        (
            "g.feelings",
            r"\b(?:hurt|hurting|kill|killing|ruin|ruining) (?:his|her|their|your|my) (?:feelings|"
            r"vibe|mood|buzz|chances|career|reputation|ego)\b",
            None,
        ),
        (
            "g.message",
            r"\bshoot (?:him|her|them|me|you|my \w+) (?:a |an )?(?:text|message|email|msg|line|"
            r"dm|call|note|quick)\b",
            None,
        ),
        ("g.hit_up", r"\bhit(?:s|ting)? (?:me|him|her|them) up\b", None),
        (
            "g.hit_with",
            r"\bhit(?:s)? me with (?:the |a |an |this |that |some |your )?(?:news|question|idea|"
            r"joke|truth|facts|update|details|pillow)\b",
            None,
        ),
        ("g.beat_at", r"\bbeat(?:s|en)? (?:me|him|her|them) (?:at|in|to|fair)\b", None),
        ("g.beaten_at", r"\bbeaten (?:at|in) \w+\b", None),
        ("g.pushed_to", r"\bpush(?:es|ed)? me (?:to|into|towards)\b", None),
        ("g.hit_by", r"\bhit by (?:a |the )?(?:car|bus|ball|truck|bike|van|train|wave)\b", None),
        (
            "g.harm_joke",
            rf"\b(?:kill|murder|strangle) {_PERSON}\b(?: \w+){{0,6}} {_JOKE}\b",
            None,
        ),
        (
            "g.suicide_terms",
            r"\bsuicide (?:prevention|awareness|rates?|statistics|stats|bomb\w*|squad|doors?|"
            r"drills?|runs?|missions?|watch|forest|sprints?)\b",
            None,
        ),
        (
            "g.live_somewhere",
            r"\bwant to live (?:in|with|here|there|near|at|on|abroad|together|alone|closer|"
            r"somewhere|like (?:that|them|her|him))\b",
            None,
        ),
        ("g.end_things_with", r"\b(?:end|ending|ended) (?:things|it) (?:with|between)\b", None),
        (
            "g.overdose_on",
            r"\boverdos(?:e|ed|ing) on (?:sugar|caffeine|coffee|chocolate|netflix|tv|anime|"
            r"content|memes|work|candy|sweets)\b",
            None,
        ),
        # Negated statements: recorded as concern.
        (
            "g.negated_want_die",
            r"\b(?:don'?t|do not|dont|never|not) (?:really |ever )?want(?:ed)? to die\b",
            "cc.negated",
        ),
        (
            "g.negated_suicidal",
            r"\b(?:i'?m|im|i am|i was|i feel) (?:\w+ )?(?:not|never) (?:\w+ )?suicidal\b",
            "cc.negated",
        ),
        (
            "g.negated_self",
            rf"\b(?:never|not|wouldn'?t|won'?t|would never|will never|could never) "
            rf"(?:ever |really )?(?:kill|hurt|harm|cut) ?{_SELF}\b",
            "cc.negated",
        ),
    )
)

# Fiction, games, lyrics and news: explicit hits become concern.
_FICTION = re.compile(
    r"\b(?:in|for) (?:my|a|the|this|our) (?:novel|story|book|script|screenplay|fic|fanfic|"
    r"fan fiction|poem|song|play|comic|manga|campaign|dnd|game|rp|roleplay|movie|film|show|"
    r"series|episode|essay)\b"
    r"|\bmy (?:character|oc|protagonist|villain|narrator|mc)\b"
    r"|\bthe (?:character|protagonist|villain|hero|heroine|narrator|main character)\b"
    r"|\b(?:chapter|scene|episode|season) \d+\b"
    r"|\b(?:writing|wrote|reading|watching|read|watched) (?:a|the|my|this) (?:story|novel|book|"
    r"scene|poem|script|song|fic|movie|film|show|article)\b"
    r"|\b(?:lyrics|storyline|fictional)\b"
    r"|\b(?:in|on) the news\b|\b(?:news|article|report|documentary) (?:said|says|about)\b"
)
_MASK = " ~ "

# Categories whose explicit hit arms the imminence cues.
_ARMING = frozenset({_C.SELF_HARM, _C.HARM_TO_OTHERS, _C.IMMINENT_DANGER})


@dataclass(frozen=True)
class PatternHit:
    rule_id: str
    category: SafetyCategory
    strength: Strength


@dataclass(frozen=True)
class PatternResult:
    hits: tuple[PatternHit, ...] = ()
    # Ids of the guards that removed text (idioms) or downgraded hits (fiction).
    guards: tuple[str, ...] = ()

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(h.rule_id for h in self.hits)

    @property
    def explicit(self) -> tuple[PatternHit, ...]:
        return tuple(h for h in self.hits if h.strength == "explicit")

    @property
    def explicit_category(self) -> SafetyCategory:
        """The most severe explicit category (`none` without explicit hits)."""
        return max_category(h.category for h in self.explicit)

    @property
    def has_concern(self) -> bool:
        return any(h.strength == "concern" for h in self.hits)


def _mask(text: str, fired: dict[str, str | None]) -> str:
    for guard_id, pattern, concern in _GUARDS:
        text, count = pattern.subn(_MASK, text)
        if count:
            fired.setdefault(guard_id, concern)
    return text


def _matches(rules: Iterable[Rule], texts: Iterable[str]) -> dict[str, Rule]:
    found: dict[str, Rule] = {}
    texts = tuple(texts)
    for rule in rules:
        if any(rule.pattern.search(text) for text in texts):
            found[rule.id] = rule
    return found


def _screen_one(message: str) -> tuple[list[PatternHit], tuple[str, ...], bool, tuple[str, ...]]:
    """(hits without cues, masked texts, fiction marker, ids of the guards that fired)."""
    fired: dict[str, str | None] = {}
    texts = tuple(_mask(text, fired) for text in variants(message))
    fiction = any(_FICTION.search(text) for text in texts)
    if fiction:
        fired.setdefault("g.fiction", None)
    hits = [
        PatternHit(
            rule.id,
            rule.category,
            "concern" if fiction and rule.strength == "explicit" else rule.strength,
        )
        for rule in _matches(RULES, texts).values()
    ]
    for concern in dict.fromkeys(c for c in fired.values() if c is not None):
        if concern not in {h.rule_id for h in hits}:
            hits.append(PatternHit(concern, _C.AMBIGUOUS_CONCERN, "concern"))
    return hits, texts, fiction, tuple(fired)


def _arms(hits: Iterable[PatternHit]) -> bool:
    return any(h.strength == "explicit" and h.category in _ARMING for h in hits)


def screen(message: str, previous: str | None = None) -> PatternResult:
    """Screen the person's latest message; `previous` is their previous message
    (only used to arm imminence cues, its own hits are not returned)."""
    hits, texts, fiction, fired = _screen_one(message)
    armed = _arms(hits) or (previous is not None and _arms(_screen_one(previous)[0]))
    if armed and not fiction:
        hits.extend(
            PatternHit(rule.id, rule.category, rule.strength)
            for rule in _matches(CUES, texts).values()
        )
    return PatternResult(hits=tuple(hits), guards=fired)
