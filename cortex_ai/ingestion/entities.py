"""Entity extraction and semantic tagging."""

import re
from typing import Optional

# Configurable known entities (users populate this for their domain)
KNOWN_ENTITIES: dict[str, list[str]] = {
    # "Canonical Name": ["alias1", "alias2", ...]
    # Example:
    # "Acme Corp": ["Acme Corp", "Acme", "ACME"],
}


def extract_entities(text: str) -> list[str]:
    """Extract entities via known-entity matching + proper noun regex."""
    found: set[str] = set()
    lower = text.lower()

    for canonical, aliases in KNOWN_ENTITIES.items():
        for alias in aliases:
            if alias.lower() in lower:
                found.add(canonical)
                break

    # Capitalized multi-word phrases (proper nouns)
    proper_nouns = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
    for noun in proper_nouns:
        if noun not in found and len(noun.split()) <= 3:
            found.add(noun)

    return list(found)


# Semantic tag patterns
_TAG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdecision\b|\bdecided\b|\bagreed\b", re.I), "decision"),
    (re.compile(r"\btask\b|\btodo\b|\baction item\b", re.I), "task"),
    (re.compile(r"\bmeeting\b|\bcall\b|\bsync\b", re.I), "meeting"),
    (re.compile(r"\bprice\b|\bcost\b|\bpayment\b|\binvoice\b|\bbudget\b", re.I), "financial"),
    (re.compile(r"\bbug\b|\bfix\b|\berror\b|\bissue\b", re.I), "technical"),
    (re.compile(r"\bidea\b|\bconcept\b|\bbrainstorm\b", re.I), "ideation"),
    (re.compile(r"\blearn\b|\blesson\b|\binsight\b", re.I), "learning"),
    (re.compile(r"\bdeadline\b|\burgent\b|\basap\b", re.I), "urgent"),
    (re.compile(r"\bfeedback\b|\breview\b", re.I), "feedback"),
    (re.compile(r"\bpersonal\b|\bfamily\b|\bkids?\b", re.I), "personal"),
    (re.compile(r"\bapi\b|\bcode\b|\bdeploy\b|\bserver\b", re.I), "engineering"),
    (re.compile(r"\bstrategy\b|\bplan\b|\broadmap\b", re.I), "strategy"),
]


def extract_semantic_tags(text: str) -> list[str]:
    """Extract semantic tags from text based on content patterns."""
    return [tag for pattern, tag in _TAG_PATTERNS if pattern.search(text)]
