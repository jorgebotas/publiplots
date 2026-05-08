"""Tests for publiplots.utils.rounding and pp.barplot(border_radius=).

The unit tests for :func:`normalize_border_radius` land in commit 2
alongside the helper; the ``pp.barplot`` integration tests land in
commit 3 when the kwarg is wired in.
"""

import pytest

from publiplots.utils.rounding import normalize_border_radius


def test_normalize_scalar():
    """Scalar int/float maps to symmetric (v, v)."""
    assert normalize_border_radius(1.5) == (1.5, 1.5)
    assert normalize_border_radius(2) == (2.0, 2.0)
    assert normalize_border_radius(0) == (0.0, 0.0)


def test_normalize_tuple():
    """2-tuple / list passes through as (top, bottom) cast to float."""
    assert normalize_border_radius((2, 0)) == (2.0, 0.0)
    assert normalize_border_radius([1.5, 0.5]) == (1.5, 0.5)
    assert normalize_border_radius((0, 0)) == (0.0, 0.0)


def test_normalize_none_is_flat():
    """None → (0.0, 0.0) — matches the rcParam default."""
    assert normalize_border_radius(None) == (0.0, 0.0)


def test_normalize_invalid_raises():
    """String, 3-tuple, dict, bool — all TypeError."""
    with pytest.raises(TypeError):
        normalize_border_radius("rounded")
    with pytest.raises(TypeError):
        normalize_border_radius((1, 2, 3))
    with pytest.raises(TypeError):
        normalize_border_radius({"top": 1, "bottom": 0})
    with pytest.raises(TypeError):
        normalize_border_radius(True)
