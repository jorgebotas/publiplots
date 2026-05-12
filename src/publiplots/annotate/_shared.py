"""Shared helpers used by every annotate strategy.

Factored out of bar_values, point_values, box_stats, bar_custom so the
custom strategies for non-bar marks can reuse them without a fifth copy.

`maybe_expand_limits` takes an explicit `axes_to_expand` list because
different strategies expand different axes:

- bar_values/bar_custom: always expand both value and cat axes.
- point_values/point_custom: expand based on anchor direction + rotation.
- box_stats/box_custom/violin_custom: expand based on anchor direction + rotation.

Each caller computes its axes_to_expand list once and passes it in.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Literal, Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._positioning import mm_to_data


def format_value(value: float, fmt: str) -> str:
    """Apply ``fmt`` to a numeric value. Handles either bare spec (``".2f"``)
    or template string (``"{:.2f}"``).
    """
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)
    return format(value, fmt)


def format_column_value(value, fmt: str) -> Optional[str]:
    """Apply ``fmt`` to a DataFrame cell. Return None for NaN / None so the
    caller can skip that record.
    """
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)
    return format(value, fmt) if fmt != "{}" else str(value)


def ensure_renderer(ax: Axes):
    """Return a valid renderer for ``ax``, drawing the canvas once if needed."""
    canvas = ax.figure.canvas
    renderer = canvas.get_renderer() if hasattr(canvas, "get_renderer") else None
    if renderer is None:
        canvas.draw()
        renderer = canvas.get_renderer()
    return renderer


_DIM_TO_FIELD = {"cat": "category", "hue": "hue_value", "hatch": "hatch_value"}


def build_column_label_table(
    records: Sequence,
    group_keys: Sequence[str],
    group_dims: Optional[Sequence[str]],
    column: str,
    data,
) -> Dict[int, object]:
    """Return ``{draw_index: cell_value}`` for each record.

    ``data.groupby(group_keys, observed=True)[column].first()`` resolves the
    per-group value. ``group_dims`` tells us which record attribute each
    group_key corresponds to (``"cat"``/``"hue"``/``"hatch"``) so we can
    build the lookup tuple without assuming positional ordering. Emits a
    ``UserWarning`` if the column varies within any group.
    """
    if column not in data.columns:
        raise KeyError(
            f"column {column!r} not found in data; available: "
            f"{list(data.columns)}"
        )
    groupby = data.groupby(list(group_keys), observed=True)
    firsts = groupby[column].first()
    nunique = groupby[column].nunique(dropna=False)
    varies = nunique[nunique > 1]
    if len(varies):
        warnings.warn(
            f"pp.annotate: column {column!r} varies within group(s) "
            f"{list(varies.index)}; using .first()",
            UserWarning,
            stacklevel=4,
        )

    dims = group_dims or ("cat",) * len(group_keys)
    table: Dict[int, object] = {}
    for rec in records:
        key_parts = [getattr(rec, _DIM_TO_FIELD[d]) for d in dims]
        key = key_parts[0] if len(key_parts) == 1 else tuple(key_parts)
        try:
            table[rec.draw_index] = firsts.loc[key]
        except KeyError:
            table[rec.draw_index] = None
    return table


_MAX_EXPAND_ITERS = 4


def maybe_expand_limits(
    ax: Axes,
    texts: List[Text],
    axes_to_expand: Sequence[Literal["x", "y"]],
    pad_mm: float,
    owner_is_publiplots: bool,
) -> None:
    """Expand the given axes so every text fits with ``pad_mm`` of breathing
    room. Iterates up to 4 times to converge on a stable layout.
    """
    if not texts or not axes_to_expand:
        return

    renderer = ensure_renderer(ax)
    for _ in range(_MAX_EXPAND_ITERS):
        ax.figure.canvas.draw()
        inv = ax.transData.inverted()
        extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]
        any_changed = False
        for ax_name in axes_to_expand:
            if _expand_axis(
                ax, extents, axis=ax_name, pad_mm=pad_mm,
                owner_is_publiplots=owner_is_publiplots,
            ):
                any_changed = True
        if not any_changed:
            break


def _expand_axis(
    ax: Axes,
    extents: list,
    axis: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> bool:
    """Expand one axis to fit extents. Return True if limits changed."""
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

    inverted = cur_lo > cur_hi
    cur_min = min(cur_lo, cur_hi)
    cur_max = max(cur_lo, cur_hi)

    if should_expand:
        new_min = min(cur_min, need_min - pad_data)
        new_max = max(cur_max, need_max + pad_data)
        if new_min == cur_min and new_max == cur_max:
            return False
        if inverted:
            set_lim(new_max, new_min)
        else:
            set_lim(new_min, new_max)
        return True
    else:
        if need_min < cur_min or need_max > cur_max:
            warnings.warn(
                "pp.annotate: labels clipped; autoscale is off on this axis",
                UserWarning,
                stacklevel=3,
            )
        return False


def compute_axes_to_expand_directional(
    anchor: str,
    orient: str,
    rotation: float,
    rotated_both: bool = False,
) -> List[Literal["x", "y"]]:
    """Pick which axes point/box strategies should expand.

    ``anchor`` in ``{"top","bottom","left","right","center"}``. ``orient``
    is ``"v"``/``"h"``; only box strategies care about it (box `left`/`right`
    labels sit at ``cat_half_width`` away from center so they expand the
    categorical axis, not the value axis). ``rotation`` turns on the
    secondary axis expansion. ``rotated_both=True`` expands both axes even
    without rotation (used by bar strategies that always expand both).
    """
    if anchor in ("top", "bottom"):
        primary = "y"
    elif anchor in ("left", "right"):
        primary = "x"
    else:
        primary = None

    rotated = rotation % 360.0 != 0.0

    if primary is None and not rotated and not rotated_both:
        return []

    axes: List[Literal["x", "y"]] = []
    if primary is not None:
        axes.append(primary)  # type: ignore[arg-type]
    if rotated or rotated_both:
        other: Literal["x", "y"] = "x" if primary == "y" else "y"
        if primary is None:
            axes.extend(["x", "y"])
        elif other not in axes:
            axes.append(other)
    return axes
