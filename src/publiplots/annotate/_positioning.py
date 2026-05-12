"""Anchor resolution and fit check. Pure math, no drawing.

`resolve_anchor` composes orientation × sign × anchor kind × errorbar
presence into a single (x, y, dx_mm, dy_mm, ha, va) tuple. The (x, y)
pair is the label's anchor on the bar edge in data coords — no offset
applied. The (dx_mm, dy_mm) pair is the mm displacement the caller
should apply via a display-space offset transform, so the pixel gap
between label and bar edge stays constant if the axis limits later
change (explicit ``set_xlim`` or implicit ``sharey=True``).

`fit_check` decides whether an already-drawn Text artist fits inside
its bar's bbox; callers use its verdict to re-anchor if needed.
`make_offset_transform` returns the ``transData + ScaledTranslation``
transform to hand to ``ax.text(..., transform=...)``.
"""
from __future__ import annotations

from typing import ClassVar, FrozenSet, Literal, Protocol, Tuple, runtime_checkable

from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.transforms import Bbox, ScaledTranslation, Transform

from publiplots.annotate._cache import BarRecord


MM_TO_INCH = 1.0 / 25.4


def mm_to_data(mm: float, ax: Axes, axis: Literal["x", "y"]) -> float:
    """Convert a millimeter displacement to a data-coord delta along `axis`.

    Uses a two-point transform: (0, 0) → (dpi * mm/25.4, 0) in display space
    is mapped back to data space; the difference is the data-coord delta.

    Callers that need the result to survive later limit changes should
    prefer `make_offset_transform` instead — this function freezes the
    conversion against the current transform.
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


def make_offset_transform(ax: Axes, dx_mm: float, dy_mm: float) -> Transform:
    """Return ``ax.transData + ScaledTranslation(dx_mm, dy_mm)``.

    The translation is expressed in ``fig.dpi_scale_trans`` (inches), so it
    stays constant in physical units across transform changes (limit edits,
    sharey-driven relim, dpi changes). Pass the result as the `transform=`
    kwarg to ``ax.text`` — the text's (x, y) then acts as a data-space
    anchor and (dx_mm, dy_mm) is the pixel-stable offset in mm.
    """
    dx_in = dx_mm * MM_TO_INCH
    dy_in = dy_mm * MM_TO_INCH
    return ax.transData + ScaledTranslation(dx_in, dy_in, ax.figure.dpi_scale_trans)


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
) -> Tuple[float, float, float, float, str, str]:
    """Return (x, y, dx_mm, dy_mm, ha, va) for a label anchor on `bar`.

    (x, y) is the label's anchor point on the bar edge in data coords —
    no offset applied. (dx_mm, dy_mm) is the mm displacement the caller
    should apply via `make_offset_transform`; the caller creates the
    text with `ax.text(x, y, ..., transform=make_offset_transform(ax, dx_mm, dy_mm))`.
    """
    left, right, bottom, top = _bar_extents(bar, orient)
    is_positive = bar.value >= 0

    if orient == "v":
        x_center = (left + right) / 2.0
        ha = "center"

        # Labels sitting ABOVE a mark use va="baseline" (not "bottom") so the
        # glyph baseline — not the bbox bottom — anchors to the offset point.
        # A digit-only string has no descenders, so va="bottom" leaves ~3px of
        # invisible descender slack between glyph ink and the bar; va="baseline"
        # eliminates that, matching the visual gap of horizontal "outside" labels.
        if anchor == "outside":
            if is_positive:
                edge = bar.err_high if bar.err_high is not None else top
                return x_center, edge, 0.0, +offset_mm, ha, "baseline"
            else:
                edge = bar.err_low if bar.err_low is not None else bottom
                return x_center, edge, 0.0, -offset_mm, ha, "top"
        if anchor == "inside":
            if is_positive:
                return x_center, top, 0.0, -offset_mm, ha, "top"
            else:
                return x_center, bottom, 0.0, +offset_mm, ha, "baseline"
        if anchor == "base":
            if ax.get_yscale() == "log":
                base = ax.get_ylim()[0]
            else:
                base = 0.0
            if is_positive:
                return x_center, base, 0.0, +offset_mm, ha, "baseline"
            else:
                return x_center, base, 0.0, -offset_mm, ha, "top"
        if anchor == "center":
            return x_center, (top + bottom) / 2.0, 0.0, 0.0, ha, "center"
        raise ValueError(f"unknown anchor: {anchor!r}")

    # orient == "h"
    y_center = (bottom + top) / 2.0
    va = "center"

    if anchor == "outside":
        if is_positive:
            edge = bar.err_high if bar.err_high is not None else right
            return edge, y_center, +offset_mm, 0.0, "left", va
        else:
            edge = bar.err_low if bar.err_low is not None else left
            return edge, y_center, -offset_mm, 0.0, "right", va
    if anchor == "inside":
        if is_positive:
            return right, y_center, -offset_mm, 0.0, "right", va
        else:
            return left, y_center, +offset_mm, 0.0, "left", va
    if anchor == "base":
        if ax.get_xscale() == "log":
            base = ax.get_xlim()[0]
        else:
            base = 0.0
        if is_positive:
            return base, y_center, +offset_mm, 0.0, "left", va
        else:
            return base, y_center, -offset_mm, 0.0, "right", va
    if anchor == "center":
        return (left + right) / 2.0, y_center, 0.0, 0.0, "center", va
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


AnchorTuple = Tuple[float, float, float, float, str, str]
#                   x,     y,     dx_mm, dy_mm, ha,  va


@runtime_checkable
class AnchorResolver(Protocol):
    """Per-mark anchor-resolution contract.

    Maps ``(record, anchor, orient, offset_mm, ax)`` to
    ``(x, y, dx_mm, dy_mm, ha, va)``. ``VALID_ANCHORS`` defines the anchor
    vocabulary for this mark type; ``DEFAULT_ANCHOR`` is the fallback when
    the caller passes ``anchor=None``.
    """
    VALID_ANCHORS: ClassVar[FrozenSet[str]]
    DEFAULT_ANCHOR: ClassVar[str]

    def resolve(
        self,
        record,
        anchor: str,
        orient: Literal["v", "h"],
        offset_mm: float,
        ax: Axes,
    ) -> AnchorTuple: ...


class BarAnchorResolver:
    """Adapter to the free-function resolve_anchor(bar, ...)."""
    VALID_ANCHORS: ClassVar[FrozenSet[str]] = frozenset({
        "outside", "inside", "base", "center",
    })
    DEFAULT_ANCHOR: ClassVar[str] = "outside"

    def resolve(self, record, anchor, orient, offset_mm, ax):
        return resolve_anchor(record, anchor, orient, offset_mm, ax)
