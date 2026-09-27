"""Hue resolution and colour derivation for wedge charts.

Colour derives from the ``hue`` level, with descendants shaded from their
parent. Two scalars control it, both expressed as a **total span** rather
than a per-ring step: the outermost ring lands at ``depth_range``, and
siblings spread over ``sibling_range`` within their parent. Spans are
depth-independent; a per-ring step puts the rim 72% toward white at five
rings and needs retuning every time a level is added.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import pandas as pd

from publiplots.annotate._builders import _is_seaborn_numeric
from publiplots.plot.wedge.geometry import WedgeNode

RGB = Tuple[float, float, float]

# Neutral grey painted when a node has no colour to draw: no palette entry
# for its hue value (a stray NaN in the hue column, or a category the
# palette does not cover). Named so the fallback is one thing to retune,
# not a literal scattered through the loop below.
_FALLBACK_COLOR = "#cccccc"


def blend(color, t: float) -> RGB:
    """Blend ``color`` toward white (``t > 0``) or black (``t < 0``)."""
    r, g, b = mcolors.to_rgb(color)
    target = 1.0 if t > 0 else 0.0
    f = abs(t)
    return (r + (target - r) * f,
            g + (target - g) * f,
            b + (target - b) * f)


def _is_numeric(series: pd.Series) -> bool:
    """Whether ``series`` counts as a continuous hue.

    Delegates to seaborn's numeric rule (``_is_seaborn_numeric``), which
    already covers the object-dtype ``all_numeric`` fallback for a column
    of ``Number``s -- matching that rule rather than
    ``is_numeric_dtype`` alone is what #262 was about. Boolean dtype is
    the one deliberate deviation: ``_is_seaborn_numeric`` treats ``bool``
    as numeric (``bool`` is a ``Number`` subclass), but a colorbar over
    ``{False, True}`` is nonsense, so a boolean hue stays categorical here.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    return _is_seaborn_numeric(series, series.dropna().unique())


def resolve_hue_values(
    data: pd.DataFrame,
    levels: List[str],
    hue: Optional[str],
    nodes: List[WedgeNode],
) -> Tuple[Dict[Tuple[Any, ...], Any], bool]:
    """Map each node path to its hue value.

    ``hue`` defaults to the root level. A categorical hue must be constant
    within a node -- otherwise the node's colour is ambiguous, and taking
    the first silently is how a figure comes to misreport a group. Naming
    a level other than the root is undefined -- every node shallower than
    that level would have no resolved value -- so it raises rather than
    silently half-colouring the chart.
    """
    column = hue if hue is not None else levels[0]
    if column in levels:
        if column != levels[0]:
            raise ValueError(
                f"hue={column!r} names level {levels.index(column)} of "
                f"{levels!r}; hue must be the outermost level "
                f"({levels[0]!r}) or a column outside the path, not a "
                f"level in between."
            )
        return {n.path: n.path[0] for n in nodes}, False

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
    hue_column: Optional[str] = None,
) -> Dict[Tuple[Any, ...], RGB]:
    """Per-node colours: palette lookup, then depth and sibling shading.

    A node with no resolved hue value, or a hue value the palette does not
    cover, is painted :data:`_FALLBACK_COLOR` rather than raising -- a
    stray ``NaN`` should not break a working figure -- but it is reported:
    one ``UserWarning`` per call names every such node, not one per node.
    """
    colors: Dict[Tuple[Any, ...], RGB] = {}
    depth_denominator = max(n_levels - 1, 1)
    missing: List[Tuple[Any, ...]] = []
    for node in nodes:
        base = palette.get(hue_values.get(node.path))
        if base is None:
            base = _FALLBACK_COLOR
            missing.append(node.path)
        depth_t = depth_range * node.depth / depth_denominator
        if node.sibling_count > 1 and node.depth > 0:
            sibling_t = sibling_range * node.sibling_index / (
                node.sibling_count - 1)
        else:
            sibling_t = 0.0
        colors[node.path] = blend(base, depth_t + sibling_t)
    if missing:
        column_note = f" (hue column {hue_column!r})" if hue_column else ""
        warnings.warn(
            f"no palette entry for the hue value at {len(missing)} "
            f"wedge(s){column_note}: {missing!r}; painted "
            f"{_FALLBACK_COLOR!r} instead. This usually means a NaN in "
            f"the hue column, or a category the palette does not cover.",
            UserWarning,
            stacklevel=2,
        )
    return colors
