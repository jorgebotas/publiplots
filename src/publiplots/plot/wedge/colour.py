"""Hue resolution and colour derivation for wedge charts.

Colour derives from the ``hue`` level, with descendants shaded from their
parent. Two scalars control it, both expressed as a **total span** rather
than a per-ring step: the outermost ring lands at ``depth_range``, and
siblings spread over ``sibling_range`` within their parent. Spans are
depth-independent; a per-ring step puts the rim 72% toward white at five
rings and needs retuning every time a level is added.
"""
from __future__ import annotations

from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import pandas as pd

from publiplots.plot.wedge.geometry import WedgeNode

RGB = Tuple[float, float, float]


def blend(color, t: float) -> RGB:
    """Blend ``color`` toward white (``t > 0``) or black (``t < 0``)."""
    r, g, b = mcolors.to_rgb(color)
    if t == 0.0:
        return (r, g, b)
    target = 1.0 if t > 0 else 0.0
    f = abs(t)
    return (r + (target - r) * f,
            g + (target - g) * f,
            b + (target - b) * f)


def _is_numeric(series: pd.Series) -> bool:
    """Mirror seaborn's ``variable_type`` numeric rule, including its
    ``all_numeric`` fallback for object-dtype columns of numbers. Matching
    the rule rather than ``is_numeric_dtype`` alone is what #262 was about.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    values = series.dropna().tolist()
    return bool(values) and all(isinstance(v, Number) for v in values)


def resolve_hue_values(
    data: pd.DataFrame,
    levels: List[str],
    hue: Optional[str],
    nodes: List[WedgeNode],
) -> Tuple[Dict[Tuple[Any, ...], Any], bool]:
    """Map each node path to its hue value.

    ``hue`` defaults to the root level. A categorical hue must be constant
    within a node -- otherwise the node's colour is ambiguous, and taking
    the first silently is how a figure comes to misreport a group.
    """
    column = hue if hue is not None else levels[0]
    if column in levels:
        depth = levels.index(column)
        return ({n.path: n.path[depth] for n in nodes if len(n.path) > depth},
                False)

    if column not in data.columns:
        raise ValueError(f"hue={column!r} is not a column of data")

    continuous = _is_numeric(data[column])
    values: Dict[Tuple[Any, ...], Any] = {}
    for node in nodes:
        mask = pd.Series(True, index=data.index)
        for i, key in enumerate(node.path):
            mask &= data[levels[i]] == key
        present = data.loc[mask, column].dropna()
        if present.empty:
            continue
        if continuous:
            values[node.path] = float(present.mean())
            continue
        distinct = list(dict.fromkeys(present.tolist()))
        if len(distinct) > 1:
            raise ValueError(
                f"hue={column!r} is not constant within the wedge "
                f"{node.path!r}: found {distinct!r}. A wedge can only take "
                f"one colour; aggregate the column or pick a hue that is "
                f"constant within each group."
            )
        values[node.path] = distinct[0]
    return values, continuous


def derive_colors(
    nodes: List[WedgeNode],
    hue_values: Dict[Tuple[Any, ...], Any],
    palette: Dict[Any, Any],
    *,
    depth_range: float,
    sibling_range: float,
    n_levels: int,
) -> Dict[Tuple[Any, ...], RGB]:
    """Per-node colours: palette lookup, then depth and sibling shading."""
    colors: Dict[Tuple[Any, ...], RGB] = {}
    depth_denominator = max(n_levels - 1, 1)
    for node in nodes:
        base = palette.get(hue_values.get(node.path))
        if base is None:
            base = "#cccccc"
        depth_t = (depth_range * node.depth / depth_denominator
                   if n_levels > 1 else 0.0)
        if node.sibling_count > 1 and node.depth > 0:
            sibling_t = sibling_range * node.sibling_index / (
                node.sibling_count - 1)
        else:
            sibling_t = 0.0
        colors[node.path] = blend(base, depth_t + sibling_t)
    return colors
