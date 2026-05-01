"""Owned-axes meta builders.

Plot functions call these to construct a strategy-specific meta object
pre-paired with the drawn artists, so the annotator doesn't have to
re-introspect the axes. The foreign-axes equivalent is `_introspect` in
`_cache.py`.

One builder per plot kind. Each takes the plot function's resolved
inputs (data, column names, palette map, errorbar spec) and returns a
`*Meta` dataclass with `owner_is_publiplots=True`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from publiplots.annotate._cache import (
    BarRecord,
    BarValueMeta,
    PointRecord,
    PointValueMeta,
    _iter_error_segments,
    _match_errorbars,
)


def _aggregate_means(
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
) -> List[Dict]:
    """Group by (categorical_axis [, hue]) and return per-group mean values.

    Row ordering: hue outer, categorical_axis inner — matches seaborn's
    dodge draw order. Errorbar extents are not computed here; callers pull
    them from the drawn errorbar artists via `_match_errorbars`.
    """
    value_col = y if categorical_axis == x else x
    cat_categories = list(data[categorical_axis].cat.categories)

    rows: List[Dict] = []
    if hue is None:
        for cat in cat_categories:
            mask = data[categorical_axis] == cat
            vals = data.loc[mask, value_col].to_numpy()
            rows.append({"mean": float(np.mean(vals))})
        return rows

    hue_categories = list(data[hue].cat.categories)
    for h in hue_categories:
        for cat in cat_categories:
            mask = (data[categorical_axis] == cat) & (data[hue] == h)
            vals = data.loc[mask, value_col].to_numpy()
            if len(vals) == 0:
                continue
            rows.append({"mean": float(np.mean(vals)), "hue_value": h})
    return rows


def _match_point_errorbars(
    ax: Axes,
    points_xy: List,
    orient: str,
    tol: float,
) -> List:
    """For each (x, y) point, find its errorbar segment.

    Match by (a) segment aligned on the categorical axis at px within `tol`,
    then (b) segment midpoint closest to py on the value axis. The second
    step disambiguates overlapping hue series that share an x-position.
    Returns [(err_low, err_high), ...] in point order.
    """
    segments = list(_iter_error_segments(ax))
    results: List = []
    for (px, py) in points_xy:
        best = None
        best_dist = float("inf")
        for xs, ys in segments:
            if orient == "v":
                if max(xs) - min(xs) >= 1e-6 or abs(xs[0] - px) > tol:
                    continue
                seg_low = float(min(ys))
                seg_high = float(max(ys))
                seg_mid = (seg_low + seg_high) / 2.0
                dist = abs(seg_mid - py)
            else:
                if max(ys) - min(ys) >= 1e-6 or abs(ys[0] - py) > tol:
                    continue
                seg_low = float(min(xs))
                seg_high = float(max(xs))
                seg_mid = (seg_low + seg_high) / 2.0
                dist = abs(seg_mid - px)
            if dist < best_dist:
                best = (seg_low, seg_high)
                best_dist = dist
        results.append(best if best is not None else (None, None))
    return results


def build_from_pointplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    errorbar: Optional[str],
) -> PointValueMeta:
    """Build a `PointValueMeta` paired with the pointplot's marker positions.

    Means come from a groupby of the raw data (hue-outer, cat-inner, matching
    seaborn's draw order). Errorbar extents are pulled from the drawn
    errorbar Line2D segments via `_match_point_errorbars`. Hue colors come
    from the resolved palette map.
    """
    orient = "v" if categorical_axis == x else "h"
    cat_categories = list(data[categorical_axis].cat.categories)

    agg = _aggregate_means(data, x=x, y=y, hue=hue,
                           categorical_axis=categorical_axis)

    # Build (xy, hue_value) pairs in the same order as agg.
    if hue is None:
        points_xy = []
        for i, cat in enumerate(cat_categories):
            pos = i  # categorical axis uses integer positions
            val = agg[i]["mean"]
            points_xy.append((pos, val) if orient == "v" else (val, pos))
        hue_values_for_points: List = [None] * len(points_xy)
    else:
        hue_categories = list(data[hue].cat.categories)
        points_xy = []
        hue_values_for_points = []
        idx = 0
        for h in hue_categories:
            for i, cat in enumerate(cat_categories):
                if idx >= len(agg):
                    break
                row = agg[idx]
                if row.get("hue_value") != h:
                    continue
                val = row["mean"]
                points_xy.append((i, val) if orient == "v" else (val, i))
                hue_values_for_points.append(h)
                idx += 1

    # Tolerance on the categorical axis: 0.5 is conservative (integer positions)
    tol = 0.5
    err_by_point = _match_point_errorbars(ax, points_xy, orient, tol)

    points: List[PointRecord] = []
    for (xy, hue_value, (err_low, err_high)) in zip(
        points_xy, hue_values_for_points, err_by_point
    ):
        hue_color = None
        if hue_value is not None and palette is not None and hue_value in palette:
            hue_color = to_rgba(palette[hue_value])
        value = xy[1] if orient == "v" else xy[0]
        points.append(PointRecord(
            xy=xy,
            value=float(value),
            err_low=err_low,
            err_high=err_high,
            hue_color=hue_color,
        ))
    return PointValueMeta(
        orient=orient,
        points=points,
        errorbar_kind=errorbar if isinstance(errorbar, str) else None,
        hue_active=hue is not None,
        owner_is_publiplots=True,
    )


def build_from_barplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    errorbar: Optional[str],
) -> BarValueMeta:
    """Build a `BarValueMeta` paired with the barplot's Rectangles.

    Means come from a groupby of the raw data; errorbar extents are pulled
    from the drawn errorbar artists via `_match_errorbars`, so they match
    seaborn exactly for every errorbar spec (`"ci"`, `"pi"`, tuples,
    callables, `None`). Records are paired with Rectangles in draw order.
    Hue colors come from the resolved palette map.
    """
    orient = "v" if categorical_axis == x else "h"

    agg = _aggregate_means(data, x=x, y=y, hue=hue,
                           categorical_axis=categorical_axis)
    rects = [
        p for p in ax.patches
        if isinstance(p, Rectangle) and p.get_width() > 0 and p.get_height() > 0
    ]
    err_by_bar = _match_errorbars(ax, rects, orient)

    bars: List[BarRecord] = []
    for rect, row, (err_low, err_high) in zip(rects, agg, err_by_bar):
        hue_color = None
        if hue is not None and palette is not None:
            key = row.get("hue_value")
            if key is not None and key in palette:
                hue_color = to_rgba(palette[key])
        if hue_color is None:
            hue_color = tuple(rect.get_facecolor())
        bars.append(BarRecord(
            patch=rect,
            value=float(row["mean"]),
            err_low=err_low,
            err_high=err_high,
            hue_color=hue_color,
        ))
    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=errorbar,
        hue_active=hue is not None,
        owner_is_publiplots=True,
    )
