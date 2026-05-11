"""Private cache contract between plot functions and annotation strategies.

`BarValueMeta` is attached to an Axes as `ax._publiplots_bar_meta` by
`pp.barplot(..., annotate=...)`. When absent (foreign axes), `_introspect`
reconstructs an equivalent meta by walking `ax.patches` and `ax.lines`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


RGBA = Tuple[float, float, float, float]


@dataclass
class BarRecord:
    patch: Rectangle
    value: float
    err_low: Optional[float]
    err_high: Optional[float]
    hue_color: Optional[RGBA]
    anchor_override: Optional[str] = None


@dataclass
class BarValueMeta:
    orient: Literal["v", "h"]
    bars: List[BarRecord]
    errorbar_kind: Optional[str]
    hue_active: bool
    owner_is_publiplots: bool


@dataclass
class PointRecord:
    """A single point's anchor + aggregated value + errorbar extents.

    Unlike BarRecord, a point has no matplotlib artist for the *mark itself*
    (seaborn merges all points of one hue into a single Line2D). We track
    the data-coord position directly and the palette color that was used.
    """
    xy: Tuple[float, float]                # (x, y) in data coords; y is the value for orient="v"
    value: float                           # aggregated estimate (mean/median)
    err_low: Optional[float]               # lower errorbar extent on the value axis
    err_high: Optional[float]              # upper errorbar extent on the value axis
    hue_color: Optional[RGBA]              # palette color assigned by pointplot


@dataclass
class PointValueMeta:
    orient: Literal["v", "h"]              # "v" = value on y-axis (common pointplot)
    points: List[PointRecord]
    errorbar_kind: Optional[str]
    hue_active: bool
    owner_is_publiplots: bool


@dataclass
class BoxStatsRecord:
    """Statistics computed for a single box.

    `center_pos` is the categorical-axis coordinate of the box's center
    (e.g. x for orient="v"). `cat_half_width` is the half-extent of the
    drawn artist on the same axis (in data coords) — used by the strategy
    to place `left`/`right`-anchored labels just past the box edge.
    `stats` is a dict mapping stat name to its value-axis coordinate,
    populated only for the stats the caller asked to label.
    """
    center_pos: float                           # x for orient="v", y for "h"
    cat_half_width: float                       # half-width on categorical axis (data coords)
    stats: dict                                 # {"median": y, "q1": y, ...}
    hue_color: Optional[RGBA]


@dataclass
class BoxStatsMeta:
    orient: Literal["v", "h"]
    boxes: List[BoxStatsRecord]
    hue_active: bool
    owner_is_publiplots: bool


def _is_bar_rect(p) -> bool:
    if not isinstance(p, Rectangle):
        return False
    return p.get_width() > 0 and p.get_height() > 0


def _iter_error_segments(ax: Axes):
    """Yield (xs, ys) arrays for every candidate errorbar segment on `ax`.

    Matplotlib/seaborn render errorbars as either:
      - Line2D artists in `ax.lines` (e.g., seaborn's pre-3.x bar errorbars), or
      - a LineCollection in `ax.collections` (matplotlib's `ax.bar(yerr=...)`
        since ~3.6 — segments packed as (2, 2) arrays).
    """
    import math as _math

    def _finite(values) -> bool:
        for v in values:
            try:
                if _math.isnan(float(v)):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    for ln in ax.lines:
        xs, ys = ln.get_xdata(), ln.get_ydata()
        if len(xs) >= 2 and _finite(xs) and _finite(ys):
            yield xs, ys
    for col in ax.collections:
        if not isinstance(col, LineCollection):
            continue
        for seg in col.get_segments():
            if len(seg) < 2:
                continue
            xs, ys = seg[:, 0], seg[:, 1]
            if _finite(xs) and _finite(ys):
                yield xs, ys


def _match_errorbars(
    ax: Axes,
    rects: List[Rectangle],
    orient: Literal["v", "h"],
) -> List[Tuple[Optional[float], Optional[float]]]:
    """For each rect, find an errorbar line whose midpoint aligns with the bar's center.

    Matplotlib/seaborn errorbars are drawn as short Line2D segments — either
    whole vertical/horizontal bars or individual cap segments. We match by
    proximity of the segment's midpoint to each rect's center on the
    categorical axis.
    """
    if not rects:
        return []
    if orient == "v":
        widths = [r.get_width() for r in rects]
        tol = 0.5 * min(widths) if widths else 0.0
    else:
        heights = [r.get_height() for r in rects]
        tol = 0.5 * min(heights) if heights else 0.0

    segments = list(_iter_error_segments(ax))

    result: List[Tuple[Optional[float], Optional[float]]] = []
    for r in rects:
        if orient == "v":
            center = r.get_x() + r.get_width() / 2.0
            # Find vertical segment whose x is near bar center (single vertical
            # bar, not the caps, which are horizontal).
            low = high = None
            for xs, ys in segments:
                if max(xs) - min(xs) < 1e-6 and abs(xs[0] - center) <= tol:
                    low = min(ys)
                    high = max(ys)
                    break
            result.append((low, high))
        else:
            center = r.get_y() + r.get_height() / 2.0
            low = high = None
            for xs, ys in segments:
                if max(ys) - min(ys) < 1e-6 and abs(ys[0] - center) <= tol:
                    low = min(xs)
                    high = max(xs)
                    break
            result.append((low, high))
    return result


def _infer_orient(rects: List[Rectangle]) -> Literal["v", "h"]:
    """All bars share width → vertical; all share height → horizontal."""
    if not rects:
        return "v"
    widths = [r.get_width() for r in rects]
    heights = [r.get_height() for r in rects]
    w_spread = max(widths) - min(widths)
    h_spread = max(heights) - min(heights)
    return "v" if w_spread <= h_spread else "h"


def _introspect(ax: Axes) -> BarValueMeta:
    """Build a BarValueMeta from an already-drawn Axes."""
    rects = [p for p in ax.patches if _is_bar_rect(p)]
    orient = _infer_orient(rects)
    err_by_bar = _match_errorbars(ax, rects, orient)
    bars: List[BarRecord] = []
    for r, (err_low, err_high) in zip(rects, err_by_bar):
        value = r.get_height() if orient == "v" else r.get_width()
        bars.append(BarRecord(
            patch=r,
            value=float(value),
            err_low=err_low,
            err_high=err_high,
            hue_color=tuple(r.get_facecolor()),
        ))
    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=None,
        hue_active=False,
        owner_is_publiplots=False,
    )
