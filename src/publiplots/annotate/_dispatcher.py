"""Public entry point: annotate(ax, kind=..., ...).

Validates inputs and dispatches to a registered strategy. Adding a new
strategy means: write the `_<kind>_strategy` function in its own file and
register it in `_STRATEGIES` below.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Union

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate.bar_values import _bar_values_strategy
from publiplots.annotate.box_stats import _box_stats_strategy
from publiplots.annotate.point_values import _point_values_strategy


_STRATEGIES: dict[str, Callable] = {
    "bar_values": _bar_values_strategy,
    "box_stats": _box_stats_strategy,
    "point_values": _point_values_strategy,
}


def annotate(
    ax: Axes,
    kind: str = "bar_values",
    *,
    fmt: str = ".2f",
    anchor: Optional[str] = None,
    offset: float = 1.0,
    color: Union[str, tuple] = "auto",
    pad: Optional[float] = None,
    rotation: float = 0.0,
    **text_kws,
) -> List[Text]:
    """Add value labels to plot marks on ``ax``.

    publiplots' flagship annotation API. Dispatches to a strategy
    selected by ``kind`` and labels each mark on ``ax`` (bars, boxes,
    or points) in place. Offsets are specified in millimetres so labels
    sit at consistent distances regardless of axes scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate. Must already have marks drawn on it
        (e.g. via :func:`publiplots.barplot`, :func:`publiplots.boxplot`,
        :func:`publiplots.pointplot`, or :func:`publiplots.lineplot`).
    kind : {'bar_values', 'box_stats', 'point_values'}, default ``'bar_values'``
        Annotation strategy to use:

        - ``'bar_values'`` — label bars with their values; anchor
          vocabulary ``{"outside", "inside", "base", "center"}``.
        - ``'box_stats'`` — label box plots with distributional
          summary statistics.
        - ``'point_values'`` — label scatter or line-plot points;
          anchor vocabulary
          ``{"top", "bottom", "left", "right", "center"}``.
    fmt : str, default ``'.2f'``
        Either a bare format spec (e.g. ``".2f"``, ``",.0f"``) or a
        format-string template containing ``{}`` (e.g.
        ``"{:,.1f}%"``).
    anchor : str, optional
        Where to place the label relative to the mark.
        Strategy-specific — see ``kind`` above. If ``None``, each
        strategy picks its own default.
    offset : float, default ``1.0``
        Gap in millimetres between the mark (or errorbar cap, when
        present) and the label, applied outward in the anchor
        direction.
    color : str or tuple, default ``'auto'``
        Label color. Accepted values:

        - ``'auto'`` — contrast-aware (composites against translucent
          fills for legibility).
        - ``'hue'`` — inherit the mark's palette color.
        - Any matplotlib color spec — used verbatim.
    pad : float, optional
        Extra margin in millimetres between the label and the axis
        edge when auto-expanding axis limits. Defaults to ``offset``
        so the geometry reads:

        ``mark → offset → label → pad → axis edge``.
    rotation : float, default ``0.0``
        Label rotation in degrees, counter-clockwise. For right-angle
        multiples (0, 90, 180, 270) the ``(ha, va)`` alignment is
        auto-remapped so the rotated label still sits at the expected
        offset from the mark, and the categorical axis is also expanded
        so rotated labels do not clip neighbouring marks. Non-right-angle
        rotations are passed through to matplotlib as-is; fine-tuning
        alignment is the caller's responsibility in that case.
    **text_kws
        Forwarded to :meth:`matplotlib.axes.Axes.text` (e.g.
        ``fontsize``, ``fontweight``).

    Returns
    -------
    list of matplotlib.text.Text
        The Text artists created, in mark order.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the registered strategies, or if
        ``offset`` / ``pad`` is negative.

    Examples
    --------
    Label bars outside the top edge:

    >>> import publiplots as pp
    >>> fig, ax = pp.subplots()
    >>> pp.barplot(data=df, x='category', y='value', ax=ax)
    >>> pp.annotate(ax, kind='bar_values', anchor='outside', fmt='.1f')

    Label bars inside, near the base, with a percent template:

    >>> pp.annotate(ax, kind='bar_values',
    ...             anchor='base', fmt='{:,.1f}%')

    Label box-plot statistics:

    >>> fig, ax = pp.subplots()
    >>> pp.boxplot(data=df, x='group', y='value', ax=ax)
    >>> pp.annotate(ax, kind='box_stats')

    Label scatter points to the right, inheriting the palette color:

    >>> fig, ax = pp.subplots()
    >>> pp.scatterplot(data=df, x='x', y='y', hue='group', ax=ax)
    >>> pp.annotate(ax, kind='point_values',
    ...             anchor='right', color='hue')
    """
    if kind not in _STRATEGIES:
        raise ValueError(
            f"unknown kind={kind!r}; known: {sorted(_STRATEGIES)}"
        )
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if pad is None:
        pad = offset
    if pad < 0:
        raise ValueError("pad must be >= 0")
    if not math.isfinite(rotation):
        raise ValueError("rotation must be a finite number of degrees")

    return _STRATEGIES[kind](
        ax, fmt=fmt, anchor=anchor, offset=offset, color=color, pad=pad,
        rotation=rotation, **text_kws,
    )
