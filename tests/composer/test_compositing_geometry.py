"""Tests for compositing/_geometry.py — pure-Python mm→pt + align×clip math.

These tests are pure unit tests; no pypdf, no cairosvg, no matplotlib.
The geometry helpers are the foundation PR 5's pdf.py builds on.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# MM2PT constant
# ---------------------------------------------------------------------------

def test_mm2pt_constant():
    from publiplots.composer.compositing._geometry import MM2PT
    assert math.isclose(MM2PT, 72.0 / 25.4, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# compute_pdf_transform — fit clip
# ---------------------------------------------------------------------------

def test_fit_square_in_square_no_scale():
    """100mm square slot, schematic is exactly 100mm square (in pt) → identity."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    slot_bbox_mm = (10.0, 20.0, 100.0, 100.0)  # x, y, w, h
    sch_size_pt = (100.0 * MM2PT, 100.0 * MM2PT)
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        slot_bbox_mm, sch_size_pt, align="center", clip="fit",
    )
    assert math.isclose(sx, 1.0, rel_tol=1e-9)
    assert math.isclose(sy, 1.0, rel_tol=1e-9)
    assert math.isclose(tx_pt, 10.0 * MM2PT, rel_tol=1e-9)
    assert math.isclose(ty_pt, 20.0 * MM2PT, rel_tol=1e-9)


def test_fit_wide_into_square_centered():
    """200pt × 100pt schematic into 100mm × 100mm slot → scale 0.5×0.5*ratio,
    centered vertically (letterbox top + bottom)."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    sch_w_pt = 200.0
    sch_h_pt = 100.0
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (0.0, 0.0, 100.0, 100.0), (sch_w_pt, sch_h_pt),
        align="center", clip="fit",
    )
    # fit: scale = min(slot_w/sch_w, slot_h/sch_h)
    expected_scale = min(slot_w_pt / sch_w_pt, slot_h_pt / sch_h_pt)
    assert math.isclose(sx, expected_scale, rel_tol=1e-9)
    assert math.isclose(sy, expected_scale, rel_tol=1e-9)
    # Centered: translation = (slot_origin) + (slot_size − scaled_sch_size) / 2
    expected_tx = 0.0 + (slot_w_pt - sch_w_pt * expected_scale) / 2.0
    expected_ty = 0.0 + (slot_h_pt - sch_h_pt * expected_scale) / 2.0
    assert math.isclose(tx_pt, expected_tx, rel_tol=1e-9)
    assert math.isclose(ty_pt, expected_ty, rel_tol=1e-9)


def test_fit_top_left_align():
    """top-left align → translation = slot origin (no centering)."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (5.0, 5.0, 100.0, 100.0), (200.0, 100.0),
        align="top-left", clip="fit",
    )
    # PDF y-axis is BOTTOM-UP; 'top-left' in screen-coords means top of slot.
    # The slot's top-y = slot_y + slot_h; aligning the schematic's top with
    # the slot's top means ty = slot_y + slot_h - sch_h * sy.
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    scale = min(slot_w_pt / 200.0, slot_h_pt / 100.0)
    assert math.isclose(tx_pt, 5.0 * MM2PT, rel_tol=1e-9)
    assert math.isclose(ty_pt, 5.0 * MM2PT + slot_h_pt - 100.0 * scale, rel_tol=1e-9)


def test_fit_bottom_right_align():
    """bottom-right align → translation = (slot_x + slot_w - sch_w*sx, slot_y)."""
    from publiplots.composer.compositing._geometry import (
        compute_pdf_transform, MM2PT
    )
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (5.0, 5.0, 100.0, 100.0), (200.0, 100.0),
        align="bottom-right", clip="fit",
    )
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    scale = min(slot_w_pt / 200.0, slot_h_pt / 100.0)
    assert math.isclose(tx_pt, 5.0 * MM2PT + slot_w_pt - 200.0 * scale, rel_tol=1e-9)
    assert math.isclose(ty_pt, 5.0 * MM2PT, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_pdf_transform — fill clip
# ---------------------------------------------------------------------------

def test_fill_uses_max_scale():
    """fill clip: scale = max(slot_w/sch_w, slot_h/sch_h)."""
    from publiplots.composer.compositing._geometry import compute_pdf_transform, MM2PT
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (0.0, 0.0, 100.0, 100.0), (200.0, 100.0),
        align="center", clip="fill",
    )
    slot_w_pt = 100.0 * MM2PT
    slot_h_pt = 100.0 * MM2PT
    expected_scale = max(slot_w_pt / 200.0, slot_h_pt / 100.0)
    assert math.isclose(sx, expected_scale, rel_tol=1e-9)
    assert math.isclose(sy, expected_scale, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_pdf_transform — stretch clip
# ---------------------------------------------------------------------------

def test_stretch_independent_axes():
    """stretch clip: sx and sy independent; align ignored (always slot origin)."""
    from publiplots.composer.compositing._geometry import compute_pdf_transform, MM2PT
    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        (5.0, 7.0, 100.0, 50.0), (200.0, 100.0),
        align="bottom-right", clip="stretch",  # align ignored
    )
    assert math.isclose(sx, 100.0 * MM2PT / 200.0, rel_tol=1e-9)
    assert math.isclose(sy, 50.0 * MM2PT / 100.0, rel_tol=1e-9)
    assert math.isclose(tx_pt, 5.0 * MM2PT, rel_tol=1e-9)
    assert math.isclose(ty_pt, 7.0 * MM2PT, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_compute_pdf_transform_zero_schematic_dim_raises():
    """A schematic with zero width or height is malformed."""
    from publiplots.composer.compositing._geometry import compute_pdf_transform
    with pytest.raises(ValueError, match=r"schematic.*zero"):
        compute_pdf_transform((0.0, 0.0, 100.0, 100.0), (0.0, 100.0),
                              align="center", clip="fit")
