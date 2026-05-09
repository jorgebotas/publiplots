"""Value-label strategy for pointplots.

Each point gets a label showing its aggregated value. Unlike bars, points
have no interior — so the anchor vocabulary is directional: top, bottom,
left, right, center. `top`/`bottom` respect the vertical errorbar cap if
present; `left`/`right` respect the horizontal cap. `center` overlays the
label on the marker (callout use case).
"""
from __future__ import annotations

import math
import warnings
from typing import List, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import PointRecord, PointValueMeta
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    mm_to_data,
    remap_alignment_for_rotation,
)


_VALID_ANCHORS = {"top", "bottom", "left", "right", "center"}
_DEFAULT_ANCHOR = "top"

# Marker radius fallback (mm) used when no errorbar cap exists in the
# label's direction. Keeps the label clear of the marker glyph.
_MARKER_CLEARANCE_MM = 1.5


def _resolve_point_anchor(
    point: PointRecord,
    anchor: str,
    offset_mm: float,
    ax: Axes,
) -> Tuple[float, float, str, str]:
    """Return (x, y, ha, va) for a label on a point.

    For top/bottom: base y on the vertical errorbar cap (err_high/err_low)
    if present; otherwise the marker y plus marker clearance.
    For left/right: same logic on the x axis.
    For center: on the marker.
    """
    px, py = point.xy
    offset_y = mm_to_data(offset_mm, ax, axis="y") if anchor in ("top", "bottom") else 0.0
    offset_x = mm_to_data(offset_mm, ax, axis="x") if anchor in ("left", "right") else 0.0
    clearance_y = mm_to_data(_MARKER_CLEARANCE_MM, ax, axis="y")
    clearance_x = mm_to_data(_MARKER_CLEARANCE_MM, ax, axis="x")

    if anchor == "top":
        edge = point.err_high if point.err_high is not None else py + clearance_y
        return px, edge + offset_y, "center", "bottom"
    if anchor == "bottom":
        edge = point.err_low if point.err_low is not None else py - clearance_y
        return px, edge - offset_y, "center", "top"
    if anchor == "right":
        # For vertical pointplots (value on y), err_high/err_low live on y,
        # so they're irrelevant for a right-anchor; use marker clearance.
        return px + clearance_x + offset_x, py, "left", "center"
    if anchor == "left":
        return px - clearance_x - offset_x, py, "right", "center"
    if anchor == "center":
        return px, py, "center", "center"
    raise ValueError(f"unreachable: unknown anchor {anchor!r}")


def _format_value(value: float, fmt: str) -> str:
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)
    return format(value, fmt)


def _ensure_renderer(ax: Axes):
    canvas = ax.figure.canvas
    renderer = canvas.get_renderer() if hasattr(canvas, "get_renderer") else None
    if renderer is None:
        canvas.draw()
        renderer = canvas.get_renderer()
    return renderer


def _point_values_strategy(
    ax: Axes,
    *,
    fmt: str,
    anchor,
    offset: float,
    color,
    pad: float,
    rotation: float = 0.0,
    **text_kws,
) -> List[Text]:
    if anchor is None:
        anchor = _DEFAULT_ANCHOR
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"point_values anchor must be one of {sorted(_VALID_ANCHORS)}; "
            f"got {anchor!r}"
        )

    # Top-level `rotation` wins over one smuggled via text_kws; pop silently.
    text_kws.pop("rotation", None)

    meta = getattr(ax, "_publiplots_point_meta", None)
    if not isinstance(meta, PointValueMeta):
        warnings.warn(
            "pp.annotate: kind='point_values' needs pp.pointplot-owned axes; "
            "call pp.pointplot(..., annotate=True) instead of annotating a "
            "foreign axes.",
            UserWarning,
            stacklevel=3,
        )
        return []
    if not meta.points:
        return []

    _ensure_renderer(ax)

    # pp.pointplot draws markers at zorder ~99-100 (double-layer style);
    # labels must sit above the markers to be visible when anchor="center".
    # Allow caller to override via text_kws.
    default_zorder = text_kws.pop("zorder", 101)

    texts: List[Text] = []
    for point in meta.points:
        if math.isnan(point.value):
            continue
        x, y, ha, va = _resolve_point_anchor(point, anchor, offset, ax)
        ha, va = remap_alignment_for_rotation(ha, va, rotation)
        # Color resolution: pointplot has no "bar fill" to composite onto,
        # so auto → rcParams text color; hue → palette color (as for bars).
        rgba = resolve_color(
            _bar_stand_in(point), color, "outside", ax,
            hue_active=meta.hue_active,
        )
        label = _format_value(point.value, fmt)
        t = ax.text(x, y, label, ha=ha, va=va, color=rgba,
                    rotation=rotation, zorder=default_zorder, **text_kws)
        texts.append(t)

    _maybe_expand_point_limits(ax, texts, anchor, pad_mm=pad,
                               owner_is_publiplots=meta.owner_is_publiplots,
                               rotation=rotation)
    return texts


class _bar_stand_in:
    """Minimal shim so resolve_color's `bar.hue_color` / edgecolor paths work
    on a PointRecord. resolve_color only reads hue_color and patch.get_facecolor.
    For the `outside` anchor used here, neither is called — but the signature
    still requires a `bar`-like object with `.patch` and `.hue_color` attrs."""

    def __init__(self, point: PointRecord):
        self.patch = _PatchShim(point.hue_color)
        self.hue_color = point.hue_color


class _PatchShim:
    def __init__(self, rgba):
        self._rgba = rgba or (0, 0, 0, 1)

    def get_facecolor(self):
        return self._rgba

    def get_edgecolor(self):
        return self._rgba


def _maybe_expand_point_limits(
    ax: Axes,
    texts: List[Text],
    anchor: str,
    pad_mm: float,
    owner_is_publiplots: bool,
    rotation: float = 0.0,
) -> None:
    if not texts:
        return

    # Which axis matches the unrotated anchor direction?
    # top/bottom → y; left/right → x; center → neither.
    if anchor in ("top", "bottom"):
        primary_axis = "y"
    elif anchor in ("left", "right"):
        primary_axis = "x"
    else:
        primary_axis = None

    rotated = rotation % 360.0 != 0.0

    if primary_axis is None and not rotated:
        return

    ax.figure.canvas.draw()
    renderer = _ensure_renderer(ax)
    inv = ax.transData.inverted()
    extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]

    axes_to_expand = []
    if primary_axis is not None:
        axes_to_expand.append(primary_axis)
    if rotated:
        other = "x" if primary_axis == "y" else "y"
        if primary_axis is None:
            axes_to_expand.extend(["x", "y"])
        elif other not in axes_to_expand:
            axes_to_expand.append(other)

    for ax_name in axes_to_expand:
        _expand_point_axis(
            ax, extents, axis=ax_name, pad_mm=pad_mm,
            owner_is_publiplots=owner_is_publiplots,
        )


def _expand_point_axis(
    ax: Axes,
    extents: list,
    axis: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> None:
    autoscale_on = (ax.get_autoscaley_on() if axis == "y"
                    else ax.get_autoscalex_on())
    should_expand = owner_is_publiplots or autoscale_on

    if axis == "y":
        need_min = min(e.y0 for e in extents)
        need_max = max(e.y1 for e in extents)
        get_lim, set_lim = ax.get_ylim, ax.set_ylim
    else:
        need_min = min(e.x0 for e in extents)
        need_max = max(e.x1 for e in extents)
        get_lim, set_lim = ax.get_xlim, ax.set_xlim

    cur_lo, cur_hi = get_lim()
    pad_data = mm_to_data(pad_mm, ax, axis=axis)

    if should_expand:
        set_lim(min(cur_lo, need_min - pad_data),
                max(cur_hi, need_max + pad_data))
    else:
        if need_min < cur_lo or need_max > cur_hi:
            warnings.warn(
                "pp.annotate: labels clipped; autoscale is off on this axis",
                UserWarning,
                stacklevel=3,
            )
