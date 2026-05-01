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
