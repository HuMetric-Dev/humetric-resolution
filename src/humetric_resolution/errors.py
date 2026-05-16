from __future__ import annotations

from dataclasses import dataclass

from humetric_core import HumetricError
from humetric_store import StoreError


@dataclass(frozen=True, slots=True)
class StoreWrapped(HumetricError):
    """Wraps a StoreError so resolution callers see a uniform ResolutionError type."""

    cause: StoreError


type ResolutionError = StoreWrapped
