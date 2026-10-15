"""Static fallback replies (design §5, bounded recovery).

When a reply still fails the output checks after one regeneration, the
turn stores one of these instead: a fixed text per pathway with the
decision's resources rendered by code, or, on the normal pathway, the
generic fallback. They are written to pass the output validator themselves
(a unit test checks every pathway in every region).

REVIEW: these texts are application copy for people who may be in crisis
and need review by a qualified person before production use.
"""

from __future__ import annotations

from app.safety.policy import SafetyDecision
from app.safety.taxonomy import Pathway

GENERIC_FALLBACK = "Sorry, I couldn't put that reply together properly. Could you say that again?"

# Pathway -> (text before the resources, text after them). Pathways without
# resources use only the first part.
FALLBACK_TEXTS: dict[Pathway, tuple[str, str]] = {
    Pathway.EMERGENCY: (
        "I'm really glad you told me, and I'm taking what you said seriously. Your safety "
        "matters most right now. Please contact emergency services now, or a crisis line if "
        "you can't:",
        "If you can, move away from anything you could use to hurt yourself or anyone else, "
        "and reach out to someone nearby you trust so you're not alone right now. I'm here, "
        "and I'll keep talking with you while you get help.",
    ),
    Pathway.CRISIS_SUPPORT: (
        "Thank you for telling me. What you're feeling matters, and so do you. Are you safe "
        "right now, and are you thinking about acting on these thoughts? You don't have to "
        "go through this alone. You can talk to someone here:",
        "If you might act on these thoughts or you're in danger right now, please contact "
        "emergency services. It could also really help to reach out to someone you trust and "
        "let them know how you're doing. I'm here and happy to keep talking.",
    ),
    Pathway.HARM_TO_OTHERS: (
        "Thank you for being honest with me about this. I can't help with hurting anyone, but "
        "I do want to help you get through what you're feeling right now. If you can, step "
        "away from the person and from anything you could use to hurt them. If anyone is in "
        "immediate danger, please contact emergency services. You can also talk it through "
        "with someone here:",
        "Reaching out to someone you trust could help too. I'm here if you want to talk about "
        "what's going on.",
    ),
    Pathway.ABUSE_SUPPORT: (
        "I'm really sorry this is happening to you. It's not your fault, and you deserve to be "
        "safe. If you're in danger right now, please contact emergency services. These can "
        "help:",
        "You don't have to face this alone, and you don't have to decide anything right now. "
        "If there's someone you trust, it may help to let them know. I'm here if you want to "
        "talk.",
    ),
    Pathway.REFUSE_HARMFUL: (
        "I can't help with that, because it could seriously hurt you or someone else. I'd "
        "still like to hear what's going on for you, though. If any part of this is about "
        "wanting to hurt yourself, you can talk to someone right now:",
        "And if you're in danger right now, please contact emergency services. I'm here to "
        "keep talking.",
    ),
    Pathway.SAFETY_CHECK_IN: (
        "That sounds really heavy, and I'm glad you told me. Can I ask you directly: are you "
        "having any thoughts of hurting yourself or ending your life? Whatever the answer, "
        "I'm here to listen.",
        "",
    ),
    Pathway.ENCOURAGE_CONNECTION: (
        "I'm really glad our chats help, and I'm here for you. I'd love for you to have people "
        "around you too, like a friend, a family member or someone else you trust, so you "
        "have support in more places than this chat. Is there anyone you've been close to "
        "that you could reach out to?",
        "",
    ),
}

assert set(FALLBACK_TEXTS) == set(Pathway) - {Pathway.NORMAL}


def fallback_reply(decision: SafetyDecision) -> str:
    """The fixed reply for the decision's pathway, with its resources."""
    if decision.pathway is Pathway.NORMAL:
        return GENERIC_FALLBACK
    before, after = FALLBACK_TEXTS[decision.pathway]
    parts = [before]
    if decision.resource_lines:
        parts.append("\n".join(decision.resource_lines))
    if after:
        parts.append(after)
    return "\n\n".join(parts)
