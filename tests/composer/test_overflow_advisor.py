"""Tests for ComposerOverflowError's suggested-scale-factor advisor."""

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerOverflowError


def test_overflow_error_carries_suggested_scale_factor():
    """The overflow advisor proposes a scale factor s such that
    requested_mm * s ≤ available_mm, telling the user how much to
    shrink the panels."""
    err = ComposerOverflowError(
        "row width 200mm exceeds canvas budget 174mm",
        requested_mm=200.0,
        available_mm=174.0,
    )
    # PR 2 adds .scale_to_fit; the formula is available / requested.
    expected = 174.0 / 200.0
    assert abs(err.scale_to_fit - expected) < 1e-9


def test_overflow_error_message_mentions_scale_factor():
    """The default __str__ includes the scale factor so users see it
    without inspecting attributes."""
    err = ComposerOverflowError(
        "row width 200mm exceeds canvas budget 174mm",
        requested_mm=200.0,
        available_mm=174.0,
    )
    msg = str(err)
    # 174/200 = 0.870 — must appear in the message at 3-digit precision.
    assert "0.87" in msg


def test_canvas_add_row_overflow_message_includes_scale_factor():
    """canvas.add_row's overflow path wraps the error with a hint;
    the user-visible message must include both the dim numbers AND
    the suggested scale factor."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ComposerOverflowError) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(100.0, 40.0)),
            pp.PanelAxes(label="B", size=(100.0, 40.0)),
        )
    msg = str(exc_info.value)
    assert "multiply" in msg.lower() or "scale" in msg.lower()
    # Suggested scale = 174 / (200 + decorations) ≈ 174/231 ≈ 0.753
    assert "0.7" in msg


def test_overflow_error_scale_factor_is_one_when_fit_exact():
    """Edge case: requested == available → scale = 1.0 (no shrinkage
    needed). Should not raise — but if it did, the factor would be 1.0."""
    err = ComposerOverflowError(
        "edge case",
        requested_mm=174.0,
        available_mm=174.0,
    )
    assert err.scale_to_fit == 1.0


def test_overflow_error_scale_factor_handles_zero_requested():
    """Defensive: if requested is 0 (degenerate), scale is +inf or 1.0
    — pick a stable, non-NaN convention. Use 1.0 (nothing to shrink)."""
    err = ComposerOverflowError(
        "degenerate",
        requested_mm=0.0,
        available_mm=174.0,
    )
    # Per implementation: if requested <= 0, return 1.0 (no shrink).
    assert err.scale_to_fit == 1.0
