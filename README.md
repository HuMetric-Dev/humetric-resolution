# humetric-resolution

Cross-source entity resolution for the Humetric workspace.

When multiple sources (GitHub, OpenAlex, NPI, LinkedIn) each contribute their
own subgraph of `Person` nodes plus native edges, the connected components
sit in isolation. This package emits `Edge(kind="same_person")` bridge edges
between probable matches across sources, so the graph branch (LightGCN) can
propagate signal across source boundaries.

## Public API

```python
from humetric_resolution import (
    MatchScore,
    ResolutionReport,
    match_score,
    resolve_all,
)
```

- `match_score(a, b) -> MatchScore` — pure function over two `Person` records.
  Returns a 0-1 score plus per-feature breakdown (name, location, skills, URL).
- `resolve_all(conn, *, threshold=0.85, target_sources=None, other_sources=None)`
  — scan the local SQLite for cross-source matches, write `same_person` edges
  (bidirectional by default), return a `ResolutionReport`. Uses a name-token
  inverted-index block to keep cost roughly linear in pair density.

The two-set `target_sources` / `other_sources` parameters express the
"privileged-hub" pattern: bridge every NPI to its OpenAlex twin without doing
the full O(N^2) cross-source sweep.

## Run

```bash
uv sync --extra dev
uv run pytest
```

## Notes / known limits

- v1 uses `difflib.SequenceMatcher` for name similarity. Jaro-Winkler is a
  better fit for first/last names; trade-off when we add a `rapidfuzz` dep.
- No model-based disambiguation: when two real people share a common name and
  city ("John Smith, Boston"), this resolver will happily bridge them. For
  scale, pair this with structured-ID anchors (ORCID, NPI registry cross-ref)
  or move to a learned matcher trained on labeled pairs.
- Doesn't currently consume `Skill` aliasing or vendor-specific URL patterns
  (e.g. a `github.com/foo` URL inside an OpenAlex bio doesn't lift the score).
  Both are easy follow-ups.
