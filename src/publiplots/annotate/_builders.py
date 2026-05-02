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

from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from matplotlib.patches import PathPatch

from publiplots.annotate._cache import (
    BarRecord,
    BarValueMeta,
    BoxStatsMeta,
    BoxStatsRecord,
    PointRecord,
    PointValueMeta,
    _iter_error_segments,
    _match_errorbars,
)


def _categories_in_draw_order(series) -> list:
    """Return the ordered category labels used by seaborn for a series.

    Accepts either a pandas Categorical (honors its explicit order) or an
    object/str Series (uses first-occurrence order, matching seaborn's
    default behavior).
    """
    if hasattr(series, "cat"):
        return list(series.cat.categories)
    # Preserve first-occurrence order without an extra import.
    seen = []
    for v in series:
        if v not in seen:
            seen.append(v)
    return seen


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
    cat_categories = _categories_in_draw_order(data[categorical_axis])

    rows: List[Dict] = []
    if hue is None:
        for cat in cat_categories:
            mask = data[categorical_axis] == cat
            vals = data.loc[mask, value_col].to_numpy()
            rows.append({"mean": float(np.mean(vals))})
        return rows

    hue_categories = _categories_in_draw_order(data[hue])
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
    cat_categories = _categories_in_draw_order(data[categorical_axis])

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
        hue_categories = _categories_in_draw_order(data[hue])
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


VALID_BOX_STATS = {"median", "q1", "q3", "whisker_low", "whisker_high", "mean"}


def _compute_box_stats(values: np.ndarray, whis: float) -> Dict[str, float]:
    """Compute box-plot statistics from raw samples.

    Matches seaborn's convention: Q1 = 25th percentile, Q3 = 75th, whiskers
    extend to the most extreme sample within `whis * IQR` of Q1/Q3.
    """
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    low_cut = q1 - whis * iqr
    high_cut = q3 + whis * iqr
    inside = values[(values >= low_cut) & (values <= high_cut)]
    whisker_low = float(inside.min()) if len(inside) else q1
    whisker_high = float(inside.max()) if len(inside) else q3
    return {
        "median": float(np.median(values)),
        "q1": q1,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "mean": float(np.mean(values)),
    }


def _aggregate_box_stats(
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    whis: float,
) -> List[Dict]:
    """Group by (categorical_axis [, hue]) and compute box stats per group.

    Row ordering matches seaborn's dodge draw order: hue-outer, cat-inner.
    """
    value_col = y if categorical_axis == x else x
    cat_categories = _categories_in_draw_order(data[categorical_axis])

    rows: List[Dict] = []
    if hue is None:
        for cat in cat_categories:
            mask = data[categorical_axis] == cat
            vals = data.loc[mask, value_col].to_numpy()
            if len(vals) == 0:
                continue
            stats = _compute_box_stats(vals, whis)
            stats["cat"] = cat
            rows.append(stats)
        return rows

    hue_categories = _categories_in_draw_order(data[hue])
    for h in hue_categories:
        for cat in cat_categories:
            mask = (data[categorical_axis] == cat) & (data[hue] == h)
            vals = data.loc[mask, value_col].to_numpy()
            if len(vals) == 0:
                continue
            stats = _compute_box_stats(vals, whis)
            stats["cat"] = cat
            stats["hue_value"] = h
            rows.append(stats)
    return rows


def _center_pos_from_artist(artist, ax: Axes, orient: str) -> float:
    """Get the categorical-axis center of a drawn artist (PathPatch or
    PolyCollection)."""
    tr = artist.get_transform() if hasattr(artist, "get_transform") else ax.transData
    # Works for both Patch (get_path) and Collection (get_paths()[0])
    if hasattr(artist, "get_path"):
        extents = artist.get_path().get_extents(tr)
    else:
        extents = artist.get_paths()[0].get_extents(tr)
    extents_data = extents.transformed(ax.transData.inverted())
    if orient == "v":
        return float((extents_data.x0 + extents_data.x1) / 2.0)
    return float((extents_data.y0 + extents_data.y1) / 2.0)


def _facecolor_of(artist) -> Tuple[float, float, float, float]:
    fc = artist.get_facecolor()
    # Collection.get_facecolor returns an array of shape (N, 4); patches return (4,)
    fc = np.asarray(fc)
    if fc.ndim == 2:
        fc = fc[0]
    return tuple(float(c) for c in fc[:4])


def _build_box_stats_meta(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    whis: float,
    artists: List,
) -> BoxStatsMeta:
    """Shared builder for pp.boxplot / pp.violinplot.

    Stats are computed from raw data (exact, matching seaborn's whis rule
    for boxplot; violinplot shows the same underlying stats). Categorical
    positions come from the drawn artists so dodge groupings are honored.
    """
    orient = "v" if categorical_axis == x else "h"
    agg = _aggregate_box_stats(data, x=x, y=y, hue=hue,
                               categorical_axis=categorical_axis, whis=whis)
    if len(artists) != len(agg):
        n = min(len(artists), len(agg))
        artists = artists[:n]
        agg = agg[:n]

    boxes: List[BoxStatsRecord] = []
    for artist, row in zip(artists, agg):
        center_pos = _center_pos_from_artist(artist, ax, orient)
        hue_color = None
        if hue is not None and palette is not None:
            key = row.get("hue_value")
            if key is not None and key in palette:
                hue_color = to_rgba(palette[key])
        if hue_color is None:
            hue_color = _facecolor_of(artist)
        boxes.append(BoxStatsRecord(
            center_pos=center_pos,
            stats={k: v for k, v in row.items() if k in VALID_BOX_STATS},
            hue_color=hue_color,
        ))
    return BoxStatsMeta(
        orient=orient,
        boxes=boxes,
        hue_active=hue is not None,
        owner_is_publiplots=True,
    )


def build_from_boxplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    whis: float,
) -> BoxStatsMeta:
    """Build a BoxStatsMeta paired with the boxplot's PathPatches."""
    patches = [p for p in ax.patches if isinstance(p, PathPatch)]
    return _build_box_stats_meta(
        ax, data, x, y, hue, categorical_axis, palette, whis, patches,
    )


def build_from_violinplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    whis: float = 1.5,
) -> BoxStatsMeta:
    """Build a BoxStatsMeta paired with the violinplot's fill collections.

    Violin draws one FillBetweenPolyCollection per violin; stats are the
    same as for boxplot (computed from raw data with matching whis rule).
    """
    from matplotlib.collections import PolyCollection
    artists = [c for c in ax.collections if isinstance(c, PolyCollection)]
    return _build_box_stats_meta(
        ax, data, x, y, hue, categorical_axis, palette, whis, artists,
    )
