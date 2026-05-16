from __future__ import annotations

import psycopg
from humetric_core import Person, Skill
from humetric_store import list_edges_from, upsert_person

from humetric_resolution import resolve_all


def _p(
    pid: str,
    source: str,
    name: str,
    *,
    location: str = "",
    skills: tuple[str, ...] = (),
) -> Person:
    return Person(
        id=pid,
        source=source,  # ty: ignore[arg-type]
        name=name,
        location=location,
        skills=tuple(Skill.of(s) for s in skills),
    )


def test_resolve_emits_bidirectional_bridge_edges(pg_conn: psycopg.Connection) -> None:
    upsert_person(
        pg_conn, _p("gh:karpathy", "github", "Andrej Karpathy", location="Stanford, CA")
    ).unwrap()
    upsert_person(
        pg_conn,
        _p(
            "oa:A123",
            "openalex",
            "Andrej Karpathy",
            location="Stanford, CA",
            skills=("deep learning",),
        ),
    ).unwrap()
    upsert_person(pg_conn, _p("npi:9999", "npi", "Andrej Karpathy")).unwrap()

    report = resolve_all(pg_conn, threshold=0.5).unwrap()

    assert report.n_pairs_compared == 3
    assert report.n_above_threshold == 3
    assert report.n_edges_written == 6

    edges = list_edges_from(pg_conn, "gh:karpathy").unwrap()
    same_person = [e for e in edges if e.kind == "same_person"]
    assert any(e.dst.startswith("oa:") for e in same_person)
    assert any(e.dst.startswith("npi:") for e in same_person)


def test_resolve_skips_same_source_pairs(pg_conn: psycopg.Connection) -> None:
    upsert_person(pg_conn, _p("gh:john1", "github", "John Smith")).unwrap()
    upsert_person(pg_conn, _p("gh:john2", "github", "John Smith")).unwrap()
    report = resolve_all(pg_conn, threshold=0.5).unwrap()
    assert report.n_pairs_compared == 0
    assert report.n_edges_written == 0


def test_resolve_below_threshold_writes_nothing(pg_conn: psycopg.Connection) -> None:
    upsert_person(pg_conn, _p("gh:a", "github", "Linus Torvalds")).unwrap()
    upsert_person(pg_conn, _p("oa:b", "openalex", "Donald Knuth")).unwrap()
    report = resolve_all(pg_conn, threshold=0.5).unwrap()
    assert report.n_pairs_compared == 0
    assert report.n_edges_written == 0


def test_resolve_respects_target_and_other_sources(pg_conn: psycopg.Connection) -> None:
    upsert_person(pg_conn, _p("gh:a", "github", "Yann LeCun")).unwrap()
    upsert_person(pg_conn, _p("oa:a", "openalex", "Yann LeCun")).unwrap()
    upsert_person(pg_conn, _p("npi:a", "npi", "Yann LeCun")).unwrap()

    report = resolve_all(
        pg_conn,
        threshold=0.5,
        target_sources=frozenset({"npi"}),
        other_sources=frozenset({"openalex"}),
        emit_reverse=False,
    ).unwrap()
    assert report.n_pairs_compared == 1
    assert report.n_above_threshold == 1
    assert report.n_edges_written == 1

    npi_edges = list_edges_from(pg_conn, "npi:a").unwrap()
    assert [(e.dst, e.kind) for e in npi_edges if e.kind == "same_person"] == [
        ("oa:a", "same_person")
    ]
    oa_edges = list_edges_from(pg_conn, "oa:a").unwrap()
    assert not [e for e in oa_edges if e.kind == "same_person"]


def test_resolve_handles_empty_db(pg_conn: psycopg.Connection) -> None:
    report = resolve_all(pg_conn).unwrap()
    assert report.n_pairs_compared == 0
    assert report.n_edges_written == 0
