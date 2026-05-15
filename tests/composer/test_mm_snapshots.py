"""mm-precision regression tests for canonical compositions.

Per spec §"Test infrastructure" line 583: assert canvas + panel mm
geometry to 0.01 mm tolerance against committed JSON snapshots in
``tests/composer/golden/mm_snapshots/``.

When intentionally changing the layout math, regenerate fixtures::

    python tools/composer/regen_fixtures.py
"""
from __future__ import annotations

import pytest

from tests.composer.golden._compositions import COMPOSITIONS
from tests.composer.golden._helpers import assert_snapshot_matches


@pytest.mark.parametrize(
    "name,build_fn",
    COMPOSITIONS,
    ids=[name for name, _ in COMPOSITIONS],
)
def test_mm_snapshot_matches(name: str, build_fn) -> None:
    """Composition ``name`` matches its committed mm snapshot."""
    canvas = build_fn()
    assert_snapshot_matches(canvas, name)
