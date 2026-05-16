from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse

from humetric_core import Person

# Suffixes that name normalization should strip before similarity comparison.
# Doctors carry "MD" / "PhD"; many GitHub bios put "(he/him)" or job titles.
_NAME_NOISE = re.compile(
    r",?\s*(m\.?d\.?|ph\.?d\.?|d\.?o\.?|m\.?p\.?h\.?|jr\.?|sr\.?|esq\.?|ma|bs|ba|"
    r"he/him|she/her|they/them)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Per-pair match breakdown. `score` is the weighted aggregate used for thresholding."""

    score: float
    name_sim: float
    location_overlap: float
    skills_overlap: float
    url_signal: float


def _normalize_name(name: str) -> str:
    """Lowercase, drop credential suffixes and punctuation, collapse whitespace."""
    s = _NAME_NOISE.sub("", name or "")
    s = _PUNCT.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip().lower()
    return s


def name_tokens(name: str) -> frozenset[str]:
    """Whitespace-split normalized name. Used by callers for blocking."""
    norm = _normalize_name(name)
    return frozenset(t for t in norm.split(" ") if len(t) > 1)


def _name_similarity(a: str, b: str) -> float:
    """SequenceMatcher.ratio over normalized names. v1 placeholder for Jaro-Winkler."""
    na = _normalize_name(a)
    nb = _normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # Token-sorted compare so "Jane Q Doe" and "Doe, Jane Q" match.
    sa = " ".join(sorted(na.split()))
    sb = " ".join(sorted(nb.split()))
    return SequenceMatcher(None, sa, sb).ratio()


def _location_tokens(loc: str) -> frozenset[str]:
    if not loc:
        return frozenset()
    parts = [p.strip().lower() for p in re.split(r"[,/;]", loc) if p.strip()]
    return frozenset(parts)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _url_signal(a: str, b: str) -> float:
    """1.0 if the raw_urls match exactly; 0.5 if same host; 0 otherwise."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    try:
        ha = urlparse(a).netloc.lower()
        hb = urlparse(b).netloc.lower()
    except ValueError:
        return 0.0
    if ha and ha == hb:
        return 0.5
    return 0.0


# Weights sum to 1. Names dominate; URL/skill/location are tie-breakers.
_W_NAME = 0.55
_W_SKILLS = 0.20
_W_LOCATION = 0.15
_W_URL = 0.10


def match_score(a: Person, b: Person) -> MatchScore:
    """Score a pair of Persons in [0, 1].

    Aggregate is a fixed linear combination. Override happens for exact-URL
    matches: those go straight to 1.0 because raw_url is an external-id-like
    field (an OpenAlex URL or an NPI registry page won't collide by accident).
    """
    name_s = _name_similarity(a.name, b.name)
    loc_s = _jaccard(_location_tokens(a.location), _location_tokens(b.location))
    skill_s = _jaccard(
        frozenset(s.normalized for s in a.skills),
        frozenset(s.normalized for s in b.skills),
    )
    url_s = _url_signal(a.raw_url, b.raw_url)

    if url_s >= 1.0:
        score = 1.0
    else:
        score = _W_NAME * name_s + _W_SKILLS * skill_s + _W_LOCATION * loc_s + _W_URL * url_s

    return MatchScore(
        score=score,
        name_sim=name_s,
        location_overlap=loc_s,
        skills_overlap=skill_s,
        url_signal=url_s,
    )
