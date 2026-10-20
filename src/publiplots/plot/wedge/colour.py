"""Hue resolution and colour derivation for wedge charts.

Colour derives from the ``hue`` level, with descendants shaded from their
parent. Two scalars control it, both expressed as a **total span** rather
than a per-ring step: the outermost ring lands at ``depth_range``, and
siblings spread over ``sibling_range`` within their parent. Spans are
depth-independent; a per-ring step of 0.18, by contrast, compounds to
4 x 0.18 = 72% toward white by the rim of a five-ring chart, and so
needs retuning every time a level is added.
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


def check_shading(depth_range: float, sibling_range: float, *,
                  owner: str) -> None:
    """Reject a depth/sibling span pair that would leave the colour range.

    The two spans add at the rim of the last sibling, so their absolute
    sum above 1 would blend past white (or black) and clip.
    """
    if abs(depth_range) + abs(sibling_range) > 1:
        raise ValueError(
            f"{owner}: depth_range={depth_range!r} and "
            f"sibling_range={sibling_range!r} together span more than the "
            f"whole colour range (|depth_range| + |sibling_range| must be "
            f"at most 1); shrink one of them.")


def resolve_hue_values(
    data: pd.DataFrame,
    levels: List[str],
    hue: Optional[str],
    nodes: List[WedgeNode],
    *,
    param_name: str = "hue",
    bucket_label: Optional[str] = None,
    owner: str = "pieplot",
    allow_level: bool = False,
) -> Tuple[Dict[Tuple[Any, ...], Any], bool]:
    """Map each node path to its hue value.

    ``hue`` defaults to the root level. The column must be a **root
    attribute**: constant within each root wedge, so that every node of
    a root takes that root's value and a root's rings share one colour
    family. A categorical column that varies within a root raises -- the
    colour would be ambiguous, and taking the first silently is how a
    figure comes to misreport a group. A continuous column takes the
    unweighted row mean per root. Resolution is one ``groupby`` over the
    root level, not a mask walk per node.

    A column naming a deeper level raises, unless ``allow_level`` is true
    (``hatch=``): a ring-``k`` column then gives every node of depth
    ``>= k`` its own ``path[k]`` and leaves shallower nodes out, so the
    pattern is inherited outward only.

    ``param_name`` is the caller's own parameter name and ``owner`` the
    caller's function name; every message raised here begins with
    ``f"{owner}: "``. The same resolution serves ``hatch=``, and a
    ``hatch`` typo reported as ``hue='zzz' is not a column of data``
    sends the reader to the wrong keyword.

    ``bucket_label``, when given, names a root key that is **not** a
    category of the data -- ``min_share``'s ``"Other"``. Several real
    categories were merged into it, so a categorical column is
    genuinely not constant across its rows; that is the merge's doing,
    not an ambiguous column, so the bucket takes its own label as its
    value instead of raising. Continuous columns are unaffected: a mean
    over the merged rows is well defined.
    """
    column = hue if hue is not None else levels[0]
    if column in levels:
        k = levels.index(column)
        if k == 0:
            return {n.path: n.path[0] for n in nodes}, False
        if allow_level:
            return {n.path: n.path[k] for n in nodes if n.depth >= k}, False
        raise ValueError(
            f"{owner}: {param_name}={column!r} names ring {k} of "
            f"{levels!r}; {param_name} must be the root ring's column "
            f"({levels[0]!r}) or a column constant within each root "
            f"wedge.")

    if column not in data.columns:
        raise ValueError(
            f"{owner}: {param_name}={column!r} is not a column of data")

    continuous = _is_numeric(data[column])
    grouped = data.dropna(subset=[levels[0]]).groupby(
        levels[0], observed=True, sort=False)[column]
    if continuous:
        # Series.mean per group, not GroupBy.mean: the groupby kernel sums
        # in a different order and differs in the last bits, which would
        # move a pie's colour and so its output bytes.
        per_root = {}
        for key, group in grouped:
            present = group.dropna()
            if not present.empty:
                per_root[key] = float(present.mean())
    else:
        counts = grouped.nunique(dropna=True)
        per_root = {}
        for key, n_distinct in counts.items():
            if bucket_label is not None and key == bucket_label:
                continue
            present = grouped.get_group(key).dropna()
            if n_distinct > 1:
                distinct = list(dict.fromkeys(present.tolist()))
                raise ValueError(
                    f"{owner}: {param_name}={column!r} is not constant "
                    f"within the wedge {(key,)!r}: found {distinct!r}. A "
                    f"wedge can only take one colour; aggregate the "
                    f"column or pick a {param_name} that is constant "
                    f"within each group.")
            if n_distinct == 1:
                per_root[key] = present.tolist()[0]
        if bucket_label is not None:
            per_root[bucket_label] = bucket_label

    values: Dict[Tuple[Any, ...], Any] = {}
    for node in nodes:
        if node.path[0] in per_root:
            values[node.path] = per_root[node.path[0]]
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
