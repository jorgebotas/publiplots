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
from typing import ClassVar, FrozenSet, List, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import PointRecord, PointValueMeta
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    AnchorTuple,
    make_offset_transform,
)
from publiplots.annotate._shared import (
    compute_axes_to_expand_directional,
    ensure_renderer as _ensure_renderer,
    format_value as _format_value,
    maybe_expand_limits,
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
) -> Tuple[float, float, float, float, str, str]:
    """Return (x, y, dx_mm, dy_mm, ha, va) for a label on a point.

    (x, y) lives in data coords on the marker (or its errorbar cap) — no
    offset applied. (dx_mm, dy_mm) is the mm displacement that keeps the
    label clear of the marker regardless of later axis-limit changes.

    For top/bottom: y-anchor is the vertical errorbar cap if present,
    otherwise the marker y itself and the marker-clearance is folded
    into dy_mm. For left/right: same idea on x. For center: on the marker.
    """
    px, py = point.xy

    if anchor == "top":
        if point.err_high is not None:
            return px, point.err_high, 0.0, +offset_mm, "center", "bottom"
        return px, py, 0.0, +_MARKER_CLEARANCE_MM + offset_mm, "center", "bottom"
    if anchor == "bottom":
        if point.err_low is not None:
            return px, point.err_low, 0.0, -offset_mm, "center", "top"
        return px, py, 0.0, -_MARKER_CLEARANCE_MM - offset_mm, "center", "top"
    if anchor == "right":
        # For vertical pointplots (value on y), err_high/err_low live on y,
        # so they're irrelevant for a right-anchor; always use marker clearance.
        return px, py, +_MARKER_CLEARANCE_MM + offset_mm, 0.0, "left", "center"
    if anchor == "left":
        return px, py, -_MARKER_CLEARANCE_MM - offset_mm, 0.0, "right", "center"
    if anchor == "center":
        return px, py, 0.0, 0.0, "center", "center"
    raise ValueError(f"unreachable: unknown anchor {anchor!r}")


class PointAnchorResolver:
    """Adapter to _resolve_point_anchor. The `orient` arg is unused (points
    have an orient on the meta for informational purposes, but the resolver
    reads the marker's own (px, py) directly)."""
    VALID_ANCHORS: ClassVar[FrozenSet[str]] = frozenset({
        "top", "bottom", "left", "right", "center",
    })
    DEFAULT_ANCHOR: ClassVar[str] = "top"

    def resolve(self, record, anchor, orient, offset_mm, ax) -> AnchorTuple:
        return _resolve_point_anchor(record, anchor, offset_mm, ax)


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
        x, y, dx_mm, dy_mm, ha, va = _resolve_point_anchor(
            point, anchor, offset, ax,
        )
        # Color resolution: pointplot has no "bar fill" to composite onto,
        # so auto → rcParams text color; hue → palette color (as for bars).
        rgba = resolve_color(
            _bar_stand_in(point), color, "outside", ax,
            hue_active=meta.hue_active,
        )
        label = _format_value(point.value, fmt)
        t = ax.text(
            x, y, label, ha=ha, va=va, color=rgba,
            rotation=rotation, zorder=default_zorder,
            transform=make_offset_transform(ax, dx_mm, dy_mm),
            **text_kws,
        )
        texts.append(t)

    maybe_expand_limits(
        ax, texts,
        axes_to_expand=compute_axes_to_expand_directional(
            anchor, meta.orient, rotation,
        ),
        pad_mm=pad, owner_is_publiplots=meta.owner_is_publiplots,
    )
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
