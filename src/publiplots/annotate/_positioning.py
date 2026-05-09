"""Anchor resolution and fit check. Pure math, no drawing.

`resolve_anchor` composes orientation × sign × anchor kind × errorbar
presence into a single (x, y, ha, va) tuple in data coordinates.
`fit_check` decides whether an already-drawn Text artist fits inside
its bar's bbox; callers use its verdict to re-anchor if needed.
`remap_alignment_for_rotation` re-maps an unrotated (ha, va) tuple so a
right-angle-rotated text's visible edge sits where the unrotated edge would.
"""
from __future__ import annotations

from typing import Literal, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from publiplots.annotate._cache import BarRecord


MM_TO_INCH = 1.0 / 25.4


def mm_to_data(mm: float, ax: Axes, axis: Literal["x", "y"]) -> float:
    """Convert a millimeter displacement to a data-coord delta along `axis`.

    Uses a two-point transform: (0, 0) → (dpi * mm/25.4, 0) in display space
    is mapped back to data space; the difference is the data-coord delta.
    """
    if mm == 0.0:
        return 0.0
    fig = ax.figure
    px = mm * fig.dpi * MM_TO_INCH
    inv = ax.transData.inverted()
    if axis == "y":
        y0 = inv.transform((0, 0))[1]
        y1 = inv.transform((0, px))[1]
        return y1 - y0
    else:
        x0 = inv.transform((0, 0))[0]
        x1 = inv.transform((px, 0))[0]
        return x1 - x0


def _bar_extents(bar: BarRecord, orient: Literal["v", "h"]):
    """Return (left, right, bottom, top) of the bar in data coords.

    Matplotlib allows bars with negative width/height — a negative-valued
    vertical bar has `get_y()=0, get_height()=-v`, so its visual bottom
    is `-v` (not 0). Normalize here so callers can reason about extents
    without worrying about sign.
    """
    r = bar.patch
    x0, x1 = r.get_x(), r.get_x() + r.get_width()
    y0, y1 = r.get_y(), r.get_y() + r.get_height()
    left, right = (x0, x1) if x0 <= x1 else (x1, x0)
    bottom, top = (y0, y1) if y0 <= y1 else (y1, y0)
    return left, right, bottom, top


def resolve_anchor(
    bar: BarRecord,
    anchor: Literal["outside", "inside", "base", "center"],
    orient: Literal["v", "h"],
    offset_mm: float,
    ax: Axes,
) -> Tuple[float, float, str, str]:
    """Return (x, y, ha, va) for a label anchor on `bar`."""
    left, right, bottom, top = _bar_extents(bar, orient)
    is_positive = bar.value >= 0

    if orient == "v":
        x_center = (left + right) / 2.0
        ha = "center"
        offset_data = mm_to_data(offset_mm, ax, axis="y")

        if anchor == "outside":
            if is_positive:
                edge = bar.err_high if bar.err_high is not None else top
                return x_center, edge + offset_data, ha, "bottom"
            else:
                edge = bar.err_low if bar.err_low is not None else bottom
                return x_center, edge - offset_data, ha, "top"
        if anchor == "inside":
            if is_positive:
                return x_center, top - offset_data, ha, "top"
            else:
                return x_center, bottom + offset_data, ha, "bottom"
        if anchor == "base":
            if ax.get_yscale() == "log":
                base = ax.get_ylim()[0]
            else:
                base = 0.0
            if is_positive:
                return x_center, base + offset_data, ha, "bottom"
            else:
                return x_center, base - offset_data, ha, "top"
        if anchor == "center":
            return x_center, (top + bottom) / 2.0, ha, "center"
        raise ValueError(f"unknown anchor: {anchor!r}")

    # orient == "h"
    y_center = (bottom + top) / 2.0
    va = "center"
    offset_data = mm_to_data(offset_mm, ax, axis="x")

    if anchor == "outside":
        if is_positive:
            edge = bar.err_high if bar.err_high is not None else right
            return edge + offset_data, y_center, "left", va
        else:
            edge = bar.err_low if bar.err_low is not None else left
            return edge - offset_data, y_center, "right", va
    if anchor == "inside":
        if is_positive:
            return right - offset_data, y_center, "right", va
        else:
            return left + offset_data, y_center, "left", va
    if anchor == "base":
        if ax.get_xscale() == "log":
            base = ax.get_xlim()[0]
        else:
            base = 0.0
        if is_positive:
            return base + offset_data, y_center, "left", va
        else:
            return base - offset_data, y_center, "right", va
    if anchor == "center":
        return (left + right) / 2.0, y_center, "center", va
    raise ValueError(f"unknown anchor: {anchor!r}")


def fit_check(
    text_artist: Text,
    bar_bbox_display: Bbox,
    orient: Literal["v", "h"],
    anchor: str,
    renderer,
) -> Literal["fits", "reanchor_outside"]:
    """Return 'fits' if text fits inside bar; 'reanchor_outside' otherwise.

    `anchor="outside"` always fits by construction.
    """
    if anchor == "outside":
        return "fits"
    tbb = text_artist.get_window_extent(renderer)
    margin_px = 1.0
    if orient == "v":
        return "fits" if tbb.height + 2 * margin_px <= bar_bbox_display.height else "reanchor_outside"
    else:
        return "fits" if tbb.width + 2 * margin_px <= bar_bbox_display.width else "reanchor_outside"


# Right-angle remap table keyed by (ha, va, quadrant) where quadrant ∈ {0,1,2,3}
# means {0°, 90°, 180°, 270°} CCW. Each (ha, va) maps the unrotated anchor to
# the alignment whose pre-rotation extension direction, when rotated CCW by
# quadrant * 90°, lands on the user's original intent direction. Derivation in
# _positioning docstring; see tests for each entry.
_ALIGN_REMAP: dict[tuple[str, str, int], tuple[str, str]] = {
    # quadrant 0 (0°): identity
    ("left", "top", 0): ("left", "top"),
    ("center", "top", 0): ("center", "top"),
    ("right", "top", 0): ("right", "top"),
    ("left", "center", 0): ("left", "center"),
    ("center", "center", 0): ("center", "center"),
    ("right", "center", 0): ("right", "center"),
    ("left", "bottom", 0): ("left", "bottom"),
    ("center", "bottom", 0): ("center", "bottom"),
    ("right", "bottom", 0): ("right", "bottom"),
    # quadrant 1 (90° CCW)
    ("left", "top", 1): ("right", "top"),
    ("center", "top", 1): ("right", "center"),
    ("right", "top", 1): ("right", "bottom"),
    ("left", "center", 1): ("center", "top"),
    ("center", "center", 1): ("center", "center"),
    ("right", "center", 1): ("center", "bottom"),
    ("left", "bottom", 1): ("left", "top"),
    ("center", "bottom", 1): ("left", "center"),
    ("right", "bottom", 1): ("left", "bottom"),
    # quadrant 2 (180°)
    ("left", "top", 2): ("right", "bottom"),
    ("center", "top", 2): ("center", "bottom"),
    ("right", "top", 2): ("left", "bottom"),
    ("left", "center", 2): ("right", "center"),
    ("center", "center", 2): ("center", "center"),
    ("right", "center", 2): ("left", "center"),
    ("left", "bottom", 2): ("right", "top"),
    ("center", "bottom", 2): ("center", "top"),
    ("right", "bottom", 2): ("left", "top"),
    # quadrant 3 (270° CCW = 90° CW)
    ("left", "top", 3): ("left", "bottom"),
    ("center", "top", 3): ("left", "center"),
    ("right", "top", 3): ("left", "top"),
    ("left", "center", 3): ("center", "bottom"),
    ("center", "center", 3): ("center", "center"),
    ("right", "center", 3): ("center", "top"),
    ("left", "bottom", 3): ("right", "bottom"),
    ("center", "bottom", 3): ("right", "center"),
    ("right", "bottom", 3): ("right", "top"),
}


def remap_alignment_for_rotation(
    ha: str,
    va: str,
    rotation: float,
) -> Tuple[str, str]:
    """Remap ``(ha, va)`` so a rotated text's visible edge lands where the
    unrotated edge would.

    For right-angle rotations (multiples of 90°) the alignment tuple is
    rotated around the text's bbox so that the user's original intent — the
    direction the text extends from its data-space anchor — is preserved in
    world space after matplotlib applies the rotation. For rotations that are
    not a multiple of 90° the input ``(ha, va)`` is returned unchanged, since
    there is no clean alignment analogue to compensate.

    Parameters
    ----------
    ha : str
        Horizontal alignment: one of ``'left'``, ``'center'``, ``'right'``.
    va : str
        Vertical alignment: one of ``'top'``, ``'center'``, ``'bottom'``,
        ``'baseline'``. ``'baseline'`` is treated as ``'bottom'``.
    rotation : float
        Rotation in degrees, counter-clockwise (matplotlib convention).
        Negative values and values ≥ 360 are normalised into ``[0, 360)``.

    Returns
    -------
    new_ha, new_va : tuple of str
        Remapped alignment. Returns the inputs unchanged when ``rotation``
        is not a right-angle multiple.

    Examples
    --------
    >>> remap_alignment_for_rotation("center", "bottom", 90)
    ('left', 'center')
    >>> remap_alignment_for_rotation("center", "bottom", 180)
    ('center', 'top')
    >>> remap_alignment_for_rotation("center", "bottom", 270)
    ('right', 'center')
    >>> remap_alignment_for_rotation("center", "bottom", 45)
    ('center', 'bottom')
    """
    va_norm = "bottom" if va == "baseline" else va
    rot = rotation % 360.0
    # Tolerance guards against float drift from negative or large inputs.
    if abs(rot - round(rot / 90.0) * 90.0) > 1e-6:
        return ha, va
    quadrant = int(round(rot / 90.0)) % 4
    key = (ha, va_norm, quadrant)
    return _ALIGN_REMAP.get(key, (ha, va))
