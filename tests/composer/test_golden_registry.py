"""Tests for the canonical composition registry.

PR 4.5 introduces tests/composer/golden/_compositions.py — a list of
(name, build_fn) tuples enumerating the compositions covered by the
mm-precision + visual-regression gates. This test asserts the registry
shape so downstream test files (test_mm_snapshots, test_png_regression)
and the CLI (tools/composer/regen_fixtures.py) can rely on it.
"""
from __future__ import annotations

import pytest


def test_registry_imports():
    """`COMPOSITIONS` is exported from _compositions.py."""
    from tests.composer.golden._compositions import COMPOSITIONS
    assert isinstance(COMPOSITIONS, list)
    assert len(COMPOSITIONS) == 4


def test_registry_entry_shape():
    """Each entry is a (name, build_fn) tuple with kebab-case name + callable."""
    from tests.composer.golden._compositions import COMPOSITIONS
    seen_names = set()
    for entry in COMPOSITIONS:
        assert isinstance(entry, tuple) and len(entry) == 2
        name, build_fn = entry
        assert isinstance(name, str)
        # kebab-case: lowercase letters + digits + hyphens only
        assert all(c.islower() or c.isdigit() or c == "-" for c in name), (
            f"name {name!r} is not kebab-case"
        )
        assert callable(build_fn)
        assert name not in seen_names, f"duplicate name {name!r}"
        seen_names.add(name)


def test_registry_build_functions_return_canvas():
    """Each build_fn returns a Canvas instance with at least one row."""
    import publiplots as pp
    from tests.composer.golden._compositions import COMPOSITIONS
    for name, build_fn in COMPOSITIONS:
        canvas = build_fn()
        assert isinstance(canvas, pp.Canvas), (
            f"{name!r} build_fn returned {type(canvas).__name__}, expected Canvas"
        )
        # Touching .figure triggers finalization — must not raise.
        _ = canvas.figure


def test_registry_build_functions_are_deterministic():
    """Calling a build_fn twice produces equivalent geometry.

    Determinism is required for snapshot stability: a build_fn that uses
    unseeded randomness or os.environ would make goldens flap.
    """
    from tests.composer.golden._compositions import COMPOSITIONS
    for name, build_fn in COMPOSITIONS:
        c1 = build_fn()
        c2 = build_fn()
        assert c1.figure_size_mm == c2.figure_size_mm, (
            f"{name!r} produced different figure_size_mm on two calls"
        )


def test_registry_includes_expected_canonical_names():
    """The four canonical compositions cloned from examples/composer/ exist."""
    from tests.composer.golden._compositions import COMPOSITIONS
    names = {name for name, _ in COMPOSITIONS}
    expected = {
        "cell-2col-simple",
        "cell-2col-multirow",
        "nature-2col-abc",
        "nature-2col-panel-grid",
    }
    assert expected.issubset(names), f"missing: {expected - names}"
