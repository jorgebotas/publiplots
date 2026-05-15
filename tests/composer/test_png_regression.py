"""PNG visual-regression tests for canonical compositions.

Per spec §"Test infrastructure" line 589: render canonical compositions
to PNG at 600 DPI and compare via ``matplotlib.testing.compare_images``
with ``tol=10`` (forgiving — catches gross regressions, not pixel diffs).

When intentionally changing the layout, regenerate fixtures::

    python tools/composer/regen_fixtures.py
"""
from __future__ import annotations

import pytest

from tests.composer.golden._compositions import COMPOSITIONS
from tests.composer.golden._helpers import assert_png_matches


@pytest.mark.parametrize(
    "name,build_fn",
    COMPOSITIONS,
    ids=[name for name, _ in COMPOSITIONS],
)
def test_png_regression_matches(name: str, build_fn) -> None:
    """Composition ``name`` renders to PNG matching its committed golden."""
    canvas = build_fn()
    assert_png_matches(canvas, name)
