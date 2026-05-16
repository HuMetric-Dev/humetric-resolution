from __future__ import annotations

from humetric_core import Person, Skill

from humetric_resolution.matcher import _normalize_name, match_score, name_tokens


def _p(
    pid: str,
    *,
    source: str = "github",
    name: str = "",
    headline: str = "",
    location: str = "",
    skills: tuple[str, ...] = (),
    raw_url: str = "",
) -> Person:
    return Person(
        id=pid,
        source=source,  # ty: ignore[arg-type]
        name=name,
        headline=headline,
        location=location,
        skills=tuple(Skill.of(s) for s in skills),
        raw_url=raw_url,
    )


def test_normalize_name_strips_credentials_and_punctuation() -> None:
    assert _normalize_name("Jane Q. Doe, MD") == "jane q doe"
    assert _normalize_name("YANN LECUN") == "yann lecun"
    assert _normalize_name("Dr. Aamer Abbas, M.D.") == "dr aamer abbas"
    assert _normalize_name("Andrej Karpathy (he/him)") == "andrej karpathy"


def test_name_tokens_drops_single_letters() -> None:
    # single-letter middle initials shouldn't pollute the blocking index
    assert name_tokens("Jane Q Doe") == frozenset({"jane", "doe"})


def test_match_score_identity_no_url_caps_at_non_url_weight_sum() -> None:
    # Without a URL signal the upper bound is _W_NAME + _W_SKILLS + _W_LOCATION = 0.90.
    a = _p("gh:a", name="Yann LeCun", location="New York, NY", skills=("deep learning",))
    b = _p(
        "oa:a",
        source="openalex",
        name="Yann LeCun",
        location="New York, NY",
        skills=("deep learning",),
    )
    assert abs(match_score(a, b).score - 0.90) < 1e-6


def test_match_score_name_only() -> None:
    a = _p("a", name="Yann LeCun")
    b = _p("b", source="openalex", name="Yann LeCun")
    ms = match_score(a, b)
    assert ms.name_sim == 1.0
    assert ms.location_overlap == 0.0
    assert ms.skills_overlap == 0.0
    assert ms.score == 0.55  # name weight only


def test_match_score_partial_name() -> None:
    a = _p("a", name="Andrej Karpathy")
    b = _p("b", source="openalex", name="A. Karpathy")
    ms = match_score(a, b)
    # Initials don't normalize identically; should still be > 0.5 via SequenceMatcher
    assert ms.name_sim > 0.5
    assert ms.score > 0.25


def test_match_score_no_overlap_is_low() -> None:
    a = _p("a", name="Linus Torvalds")
    b = _p("b", source="openalex", name="Donald Knuth")
    assert match_score(a, b).score < 0.3


def test_match_score_url_exact_overrides_to_one() -> None:
    a = _p("a", name="X", raw_url="https://orcid.org/0000-0001-2345-6789")
    b = _p("b", source="openalex", name="Y", raw_url="https://orcid.org/0000-0001-2345-6789")
    assert match_score(a, b).score == 1.0


def test_match_score_url_host_partial_credit() -> None:
    a = _p("a", name="Andrej Karpathy", raw_url="https://github.com/karpathy")
    b = _p(
        "b",
        source="openalex",
        name="Andrej Karpathy",
        raw_url="https://github.com/someone-else",
    )
    ms = match_score(a, b)
    assert ms.url_signal == 0.5
    assert ms.name_sim == 1.0
    # 0.55 * 1 + 0.10 * 0.5 = 0.60
    assert abs(ms.score - 0.60) < 1e-6


def test_match_score_token_order_invariant() -> None:
    a = _p("a", name="Jane Q Doe")
    b = _p("b", source="openalex", name="Doe, Jane Q")
    assert match_score(a, b).name_sim == 1.0


def test_match_score_skills_and_location_add_signal() -> None:
    a = _p(
        "a",
        name="Sami Aasar",
        location="Fishers, IN",
        skills=("internal medicine", "interventional cardiology"),
    )
    b = _p(
        "b",
        source="openalex",
        name="Sami Aasar",
        location="Indianapolis, IN",
        skills=("interventional cardiology", "coronary intervention"),
    )
    ms = match_score(a, b)
    assert ms.name_sim == 1.0
    assert ms.location_overlap > 0  # "IN" overlap
    assert ms.skills_overlap > 0  # "interventional cardiology" overlap
    assert ms.score > 0.6
