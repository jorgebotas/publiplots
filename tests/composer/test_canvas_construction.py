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


def test_full_journal_preset_set_in_pr2():
    """PR 2 ships Cell / Nature / Nat Methods / Science presets plus
    the 'custom' escape hatch. PR 1 only had 'custom'."""
    expected = {
        "custom",
        "cell-1col", "cell-1.5col", "cell-2col",
        "nature-1col", "nature-1.5col", "nature-2col",
        "nature-methods-1col", "nature-methods-1.5col", "nature-methods-2col",
        "science-1col", "science-1.5col", "science-2col",
    }
    assert set(PRESETS.keys()) == expected


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


# ---------------------------------------------------------------------------
# Canvas construction — PR 1 only supports preset='custom' with width=
# ---------------------------------------------------------------------------
import publiplots as pp


def test_canvas_construction_custom_preset_with_width():
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.width_mm == 174.0


def test_canvas_construction_custom_requires_width():
    with pytest.raises(ValueError, match="width"):
        pp.Canvas("custom")


def test_canvas_construction_unknown_preset_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        pp.Canvas("not-a-preset", width=100.0)


def test_canvas_figure_attribute_is_none_until_finalize():
    """A canvas before any add_row exposes .figure as None — the
    matplotlib Figure is created lazily when add_row is called.
    Rationale: a canvas with zero panels has no defined height."""
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.figure is None


def test_canvas_figure_size_mm_is_none_until_finalize():
    canvas = pp.Canvas("custom", width=174.0)
    assert canvas.figure_size_mm is None


# ---------------------------------------------------------------------------
# PR 3: ComposerAlignmentError — subclass of ComposerError
# ---------------------------------------------------------------------------

def test_composer_alignment_error_is_composer_error_subclass():
    """ComposerAlignmentError inherits from ComposerError so callers can
    catch all composer errors with one except clause."""
    from publiplots.composer.exceptions import (
        ComposerError,
        ComposerAlignmentError,
    )
    assert issubclass(ComposerAlignmentError, ComposerError)


def test_composer_alignment_error_carries_panels_and_edge():
    """ComposerAlignmentError exposes which panels and edge failed
    so callers can format helpful messages without re-parsing the str."""
    from publiplots.composer.exceptions import ComposerAlignmentError
    err = ComposerAlignmentError(
        "alignment shift would push panel A's left edge outside its slot",
        panels=("A", "B"),
        edge="left",
    )
    assert err.panels == ("A", "B")
    assert err.edge == "left"
    assert "A" in str(err)
