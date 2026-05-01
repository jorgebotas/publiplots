"""Private cache contract between plot functions and annotation strategies.

`BarValueMeta` is attached to an Axes as `ax._publiplots_bar_meta` by
`pp.barplot(..., annotate=...)`. When absent (foreign axes), `_introspect`
reconstructs an equivalent meta by walking `ax.patches` and `ax.lines`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class BarValueMeta:
    orient: Literal["v", "h"]
    bars: List[BarRecord]
    errorbar_kind: Optional[str]
    hue_active: bool
    owner_is_publiplots: bool


def _is_bar_rect(p) -> bool:
    if not isinstance(p, Rectangle):
        return False
    w, h = p.get_width(), p.get_height()
    if w <= 0 or h <= 0:
        return False
    # Matplotlib's axes frame and legend patches are Rectangles too; filter by
    # checking the patch is attached to ax.patches (caller iterates that list)
    # and that the bar is anchored on the data axis. Frame rectangles live in
    # ax.spines, not ax.patches, so iterating ax.patches already excludes them.
    return True


def _iter_error_segments(ax: Axes):
    """Yield (xs, ys) arrays for every candidate errorbar segment on `ax`.

    Matplotlib/seaborn render errorbars as either:
      - Line2D artists in `ax.lines` (e.g., seaborn's pre-3.x bar errorbars), or
      - a LineCollection in `ax.collections` (matplotlib's `ax.bar(yerr=...)`
        since ~3.6 — segments packed as (2, 2) arrays).
    """
    for ln in ax.lines:
        xs, ys = ln.get_xdata(), ln.get_ydata()
        if len(xs) >= 2:
            yield xs, ys
    for col in ax.collections:
        if not isinstance(col, LineCollection):
            continue
        for seg in col.get_segments():
            if len(seg) < 2:
                continue
            yield seg[:, 0], seg[:, 1]


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
