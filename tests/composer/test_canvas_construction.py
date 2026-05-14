"""Construction tests for pp.Canvas — exceptions, presets, validation."""

import pytest

import publiplots as pp
from publiplots.composer.exceptions import (
    ComposerError,
    ComposerOverflowError,
)


def test_composer_error_is_value_error_subclass():
    """ComposerError is the base class for all composer-specific errors;
    inheriting from ValueError lets users catch with the standard idiom."""
    assert issubclass(ComposerError, ValueError)


def test_composer_overflow_error_is_composer_error():
    assert issubclass(ComposerOverflowError, ComposerError)


def test_composer_overflow_error_carries_offending_dim():
    """ComposerOverflowError exposes the requested + available dims so
    callers (and Task 2's row-width validator) can format helpful
    messages without re-parsing the str."""
    err = ComposerOverflowError(
        "row 0 width 200mm exceeds canvas budget 174mm",
        requested_mm=200.0,
        available_mm=174.0,
    )
    assert err.requested_mm == 200.0
    assert err.available_mm == 174.0
    assert "200" in str(err)
    assert "174" in str(err)


# ---------------------------------------------------------------------------
# presets.py — only 'custom' in PR 1; journal presets land in PR 2
# ---------------------------------------------------------------------------
from publiplots.composer.presets import PRESETS, resolve_preset


def test_only_custom_preset_in_pr1():
    """PR 1 ships ONLY the 'custom' preset. Journal presets (cell, nature,
    nature-methods, science) land in PR 2. This test will be UPDATED by
    PR 2; in PR 1 it pins the preset list to exactly {'custom'} so we
    don't accidentally ship a half-baked Cell preset."""
    assert set(PRESETS.keys()) == {"custom"}


def test_resolve_preset_custom_requires_width():
    """'custom' has no default width — caller must supply ``width=``."""
    with pytest.raises(ValueError, match="width"):
        resolve_preset("custom", width=None)


def test_resolve_preset_custom_returns_width():
    p = resolve_preset("custom", width=174.0)
    assert p["width_mm"] == 174.0


def test_resolve_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        resolve_preset("not-a-preset", width=100.0)


def test_resolve_preset_rejects_non_positive_width():
    with pytest.raises(ValueError, match="positive"):
        resolve_preset("custom", width=0.0)
    with pytest.raises(ValueError, match="positive"):
        resolve_preset("custom", width=-10.0)
