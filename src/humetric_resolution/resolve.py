from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import psycopg
from humetric_core import Edge, Err, Ok, Person, Result
from humetric_store import list_persons, upsert_edge

from humetric_resolution.errors import ResolutionError, StoreWrapped
from humetric_resolution.matcher import MatchScore, match_score, name_tokens


@dataclass(frozen=True, slots=True)
class ResolutionReport:
    n_pairs_compared: int
    n_above_threshold: int
    n_edges_written: int


def _load_all(
    conn: psycopg.Connection,
    source: str | None = None,
    batch: int = 1000,
) -> Result[list[Person], ResolutionError]:
    out: list[Person] = []
    offset = 0
    while True:
        r = list_persons(conn, source=source, limit=batch, offset=offset)
        if isinstance(r, Err):
            return Err(StoreWrapped(cause=r.error))
        page = r.value
        if not page:
            break
        out.extend(page)
        if len(page) < batch:
            break
        offset += batch
    return Ok(out)


def resolve_all(
    conn: psycopg.Connection,
    *,
    threshold: float = 0.85,
    target_sources: frozenset[str] | None = None,
    other_sources: frozenset[str] | None = None,
    emit_reverse: bool = True,
) -> Result[ResolutionReport, ResolutionError]:
    """Find cross-source matches and write `same_person` bridge edges.

    Pairs are filtered to cross-source only (no same-source matches — those
    sources already have native edges and same-person duplicates within one
    source are out of scope here).

    `target_sources` / `other_sources` restrict which side of each pair is
    drawn from which set. The privileged-hub pattern is e.g.
    `target_sources={"npi"}, other_sources={"openalex"}` — bridge every NPI
    to its OpenAlex twin (one-way). Leaving both None resolves all
    cross-source pairs.

    `emit_reverse` controls whether the symmetric edge (dst -> src) is also
    written, so the graph branch's message-passing benefits both endpoints.
    """
    all_persons_r = _load_all(conn)
    if isinstance(all_persons_r, Err):
        return all_persons_r
    persons = all_persons_r.value
    if not persons:
        return Ok(ResolutionReport(0, 0, 0))

    # Name-token inverted index: token -> set of person row indices.
    # This is the cheap blocking step that turns the naive O(N^2) into
    # O(sum of bucket sizes squared), usually 10-100x smaller.
    by_token: dict[str, list[int]] = defaultdict(list)
    tokens_per_person: list[frozenset[str]] = []
    for i, p in enumerate(persons):
        toks = name_tokens(p.name)
        tokens_per_person.append(toks)
        for t in toks:
            by_token[t].append(i)

    seen_pairs: set[tuple[int, int]] = set()
    above: list[tuple[int, int, MatchScore]] = []
    pairs_compared = 0

    for i, p in enumerate(persons):
        if target_sources is not None and p.source not in target_sources:
            continue
        # Collect candidate indices that share at least one name token.
        candidates: set[int] = set()
        for t in tokens_per_person[i]:
            candidates.update(by_token[t])
        for j in candidates:
            if j == i:
                continue
            other = persons[j]
            if other.source == p.source:
                continue
            if other_sources is not None and other.source not in other_sources:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pairs_compared += 1
            ms = match_score(p, other)
            if ms.score >= threshold:
                above.append((i, j, ms))

    n_written = 0
    for i, j, _ms in above:
        a = persons[i]
        b = persons[j]
        w_r = upsert_edge(conn, Edge(src=a.id, dst=b.id, kind="same_person"))
        if isinstance(w_r, Err):
            return Err(StoreWrapped(cause=w_r.error))
        n_written += 1
        if emit_reverse:
            w_r = upsert_edge(conn, Edge(src=b.id, dst=a.id, kind="same_person"))
            if isinstance(w_r, Err):
                return Err(StoreWrapped(cause=w_r.error))
            n_written += 1

    return Ok(
        ResolutionReport(
            n_pairs_compared=pairs_compared,
            n_above_threshold=len(above),
            n_edges_written=n_written,
        )
    )
