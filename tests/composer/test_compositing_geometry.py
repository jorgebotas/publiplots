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


# ---------------------------------------------------------------------------
# _resolve_svg_units — viewBox + width/height unit detection
# ---------------------------------------------------------------------------

# matplotlib's SVG output uses pt-based viewBox with width/height in pt.
# Inkscape uses mm. Illustrator + browsers use px or unitless.
# Tests cover all 4 cases + missing viewBox + relative-unit reject +
# disagreement warning.

import pytest as _pytest  # noqa: E402  alias to avoid shadowing scope

lxml_etree = _pytest.importorskip("lxml.etree")


def _parse_svg(text):
    """Helper: parse an SVG string and return the root element."""
    return lxml_etree.fromstring(text.encode("utf-8"))


def test_resolve_svg_units_pt_viewbox_matplotlib():
    """matplotlib emits viewBox in pt with width/height in pt: mm-per-user-unit
    = 25.4/72 ≈ 0.3528 (since user unit == pt).
    """
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="288pt" height="144pt" viewBox="0 0 288 144"/>'
    )
    vb_x, vb_y, vb_w, vb_h, mm_per_uu = _resolve_svg_units(root)
    assert math.isclose(vb_x, 0.0)
    assert math.isclose(vb_y, 0.0)
    assert math.isclose(vb_w, 288.0)
    assert math.isclose(vb_h, 144.0)
    assert math.isclose(mm_per_uu, 25.4 / 72.0, rel_tol=1e-6)


def test_resolve_svg_units_mm_viewbox_inkscape():
    """Inkscape emits viewBox in user units that match mm: width/height in mm
    same as viewBox dims → mm-per-user-unit = 1.0.
    """
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30"/>'
    )
    vb_x, vb_y, vb_w, vb_h, mm_per_uu = _resolve_svg_units(root)
    assert math.isclose(vb_w, 40.0)
    assert math.isclose(vb_h, 30.0)
    assert math.isclose(mm_per_uu, 1.0, rel_tol=1e-9)


def test_resolve_svg_units_px_viewbox_illustrator():
    """Illustrator-style SVG: viewBox in px, width/height in px @ 96 DPI:
    mm-per-user-unit = 25.4/96 ≈ 0.2646.
    """
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="96px" height="48px" viewBox="0 0 96 48"/>'
    )
    _, _, _, _, mm_per_uu = _resolve_svg_units(root)
    assert math.isclose(mm_per_uu, 25.4 / 96.0, rel_tol=1e-6)


def test_resolve_svg_units_unitless_assumes_px_at_96dpi():
    """Unitless width/height (browser SVGs) → assume px @ 96 DPI."""
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="200" height="100" viewBox="0 0 200 100"/>'
    )
    _, _, _, _, mm_per_uu = _resolve_svg_units(root)
    assert math.isclose(mm_per_uu, 25.4 / 96.0, rel_tol=1e-6)


def test_resolve_svg_units_no_width_height_assumes_px_at_96dpi():
    """No width/height attributes (only viewBox) → also fall back to px@96."""
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50"/>'
    )
    _, _, _, _, mm_per_uu = _resolve_svg_units(root)
    assert math.isclose(mm_per_uu, 25.4 / 96.0, rel_tol=1e-6)


def test_resolve_svg_units_missing_viewbox_raises():
    """No viewBox → ComposerVectorError (out-of-scope-for-PR6a authoring path)."""
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    from publiplots.composer.exceptions import ComposerVectorError
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"/>'
    )
    with pytest.raises(ComposerVectorError, match=r"viewBox"):
        _resolve_svg_units(root)


def test_resolve_svg_units_relative_unit_em_raises():
    """em on width/height → ComposerVectorError (font context unavailable)."""
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    from publiplots.composer.exceptions import ComposerVectorError
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="10em" height="5em" viewBox="0 0 100 50"/>'
    )
    with pytest.raises(ComposerVectorError, match=r"relative units"):
        _resolve_svg_units(root)


def test_resolve_svg_units_relative_unit_percent_raises():
    """% on width/height → ComposerVectorError."""
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    from publiplots.composer.exceptions import ComposerVectorError
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="100%" height="50%" viewBox="0 0 100 50"/>'
    )
    with pytest.raises(ComposerVectorError, match=r"relative units"):
        _resolve_svg_units(root)


def test_resolve_svg_units_nonzero_origin_viewbox():
    """A non-zero-origin viewBox propagates vb_x/vb_y."""
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="100mm" height="50mm" viewBox="10 -5 100 50"/>'
    )
    vb_x, vb_y, vb_w, vb_h, mm_per_uu = _resolve_svg_units(root)
    assert math.isclose(vb_x, 10.0)
    assert math.isclose(vb_y, -5.0)
    assert math.isclose(vb_w, 100.0)
    assert math.isclose(vb_h, 50.0)
    assert math.isclose(mm_per_uu, 1.0)


def test_resolve_svg_units_disagreement_warns_prefers_width():
    """If width/height units yield mm-per-user-unit values that disagree
    by > 1%, emit UserWarning and prefer width.
    """
    import warnings as _w
    from publiplots.composer.compositing._geometry import _resolve_svg_units
    # width says 100mm-for-100uu → 1.0 mm/uu; height says 50pt-for-50uu →
    # 25.4/72 ≈ 0.3528 mm/uu. Big disagreement.
    root = _parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="100mm" height="50pt" viewBox="0 0 100 50"/>'
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        _, _, _, _, mm_per_uu = _resolve_svg_units(root)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "expected disagreement UserWarning, got none"
    # Width preferred → 1.0
    assert math.isclose(mm_per_uu, 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# compute_svg_transform — top-down y-axis variant
# ---------------------------------------------------------------------------

def test_compute_svg_transform_fit_square_in_square_no_scale():
    """100mm slot, schematic 100mm → identity scale; centered."""
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, tx, ty = compute_svg_transform(
        (10.0, 20.0, 100.0, 100.0),  # slot mm
        (100.0, 100.0),               # schematic mm
        canvas_mm_per_user_unit=1.0,  # 1 user unit == 1 mm
        canvas_vb_origin=(0.0, 0.0),
        align="center",
        clip="fit",
    )
    assert math.isclose(sx, 1.0)
    assert math.isclose(sy, 1.0)
    assert math.isclose(tx, 10.0)
    assert math.isclose(ty, 20.0)


def test_compute_svg_transform_top_left_align_y_axis_inverted():
    """SVG y-axis is TOP-DOWN: 'top'-words → 0 (NOT full slack like PDF).

    100mm slot @ origin (0,0). 200×100 schematic. fit-mode → scale 0.5.
    Scaled schematic is 100×50. 'top-left' → tx=0, ty=0 (top of slot ==
    schematic top in top-down y).
    """
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, tx, ty = compute_svg_transform(
        (5.0, 5.0, 100.0, 100.0), (200.0, 100.0),
        canvas_mm_per_user_unit=1.0,
        canvas_vb_origin=(0.0, 0.0),
        align="top-left", clip="fit",
    )
    assert math.isclose(sx, 0.5)
    assert math.isclose(sy, 0.5)
    assert math.isclose(tx, 5.0)
    assert math.isclose(ty, 5.0)


def test_compute_svg_transform_bottom_right_align_y_axis_inverted():
    """SVG y-axis is TOP-DOWN: 'bottom'-words → full slack.

    100mm slot. Scaled 100×50 schematic. 'bottom-right': tx=slot_w-100=0,
    ty=slot_h-50=50.
    """
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, tx, ty = compute_svg_transform(
        (5.0, 5.0, 100.0, 100.0), (200.0, 100.0),
        canvas_mm_per_user_unit=1.0,
        canvas_vb_origin=(0.0, 0.0),
        align="bottom-right", clip="fit",
    )
    # 5+100-100=5
    assert math.isclose(tx, 5.0)
    # 5+100-50=55
    assert math.isclose(ty, 55.0)


def test_compute_svg_transform_units_in_user_space_not_mm():
    """When mm-per-uu != 1.0, slot translations are in user-unit space."""
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    # Canvas: 1 user unit = 25.4/72 mm (matplotlib pt-based viewBox).
    mm_per_uu = 25.4 / 72.0
    sx, sy, tx, ty = compute_svg_transform(
        (10.0, 20.0, 50.0, 50.0),  # slot in mm
        (50.0, 50.0),               # schematic in mm
        canvas_mm_per_user_unit=mm_per_uu,
        canvas_vb_origin=(0.0, 0.0),
        align="center", clip="fit",
    )
    # slot_x in user units = 10 mm / (25.4/72) mm/uu = 28.346 uu
    assert math.isclose(tx, 10.0 / mm_per_uu, rel_tol=1e-9)
    assert math.isclose(ty, 20.0 / mm_per_uu, rel_tol=1e-9)
    # scale: schematic_size_in_uu = 50 / mm_per_uu;
    # slot_size_in_uu = 50 / mm_per_uu; ratio = 1.
    assert math.isclose(sx, 1.0, rel_tol=1e-9)
    assert math.isclose(sy, 1.0, rel_tol=1e-9)


def test_compute_svg_transform_canvas_vb_origin_offsets_translation():
    """Canvas viewBox origin offset adds to slot translations."""
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, tx, ty = compute_svg_transform(
        (10.0, 20.0, 50.0, 50.0), (50.0, 50.0),
        canvas_mm_per_user_unit=1.0,
        canvas_vb_origin=(100.0, -10.0),
        align="center", clip="fit",
    )
    assert math.isclose(tx, 10.0 + 100.0)
    assert math.isclose(ty, 20.0 + -10.0)


def test_compute_svg_transform_stretch_independent_axes():
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, tx, ty = compute_svg_transform(
        (5.0, 7.0, 100.0, 50.0), (200.0, 100.0),
        canvas_mm_per_user_unit=1.0,
        canvas_vb_origin=(0.0, 0.0),
        align="bottom-right", clip="stretch",
    )
    assert math.isclose(sx, 100.0 / 200.0)
    assert math.isclose(sy, 50.0 / 100.0)
    assert math.isclose(tx, 5.0)
    assert math.isclose(ty, 7.0)


def test_compute_svg_transform_fill_uses_max_scale():
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, _, _ = compute_svg_transform(
        (0.0, 0.0, 100.0, 100.0), (200.0, 100.0),
        canvas_mm_per_user_unit=1.0,
        canvas_vb_origin=(0.0, 0.0),
        align="center", clip="fill",
    )
    expected = max(100.0 / 200.0, 100.0 / 100.0)
    assert math.isclose(sx, expected)
    assert math.isclose(sy, expected)


def test_compute_svg_transform_invalid_align_raises():
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    with pytest.raises(ValueError, match=r"align"):
        compute_svg_transform(
            (0.0, 0.0, 100.0, 100.0), (100.0, 100.0),
            canvas_mm_per_user_unit=1.0,
            canvas_vb_origin=(0.0, 0.0),
            align="middle",  # typo
            clip="fit",
        )


def test_compute_svg_transform_invalid_clip_raises():
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    with pytest.raises(ValueError, match=r"clip"):
        compute_svg_transform(
            (0.0, 0.0, 100.0, 100.0), (100.0, 100.0),
            canvas_mm_per_user_unit=1.0,
            canvas_vb_origin=(0.0, 0.0),
            align="center", clip="cover",
        )


def test_compute_svg_transform_zero_schematic_raises():
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    with pytest.raises(ValueError, match=r"schematic.*zero"):
        compute_svg_transform(
            (0.0, 0.0, 100.0, 100.0), (0.0, 100.0),
            canvas_mm_per_user_unit=1.0,
            canvas_vb_origin=(0.0, 0.0),
            align="center", clip="fit",
        )


@pytest.mark.parametrize("align", [
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
])
@pytest.mark.parametrize("clip", ["fit", "fill", "stretch"])
def test_compute_svg_transform_all_align_clip_combinations(align, clip):
    """All 9 × 3 align × clip combos return finite floats."""
    from publiplots.composer.compositing._geometry import (
        compute_svg_transform,
    )
    sx, sy, tx, ty = compute_svg_transform(
        (10.0, 20.0, 80.0, 60.0), (200.0, 100.0),
        canvas_mm_per_user_unit=1.0,
        canvas_vb_origin=(0.0, 0.0),
        align=align, clip=clip,
    )
    assert all(math.isfinite(v) for v in (sx, sy, tx, ty))
