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

from publiplots.annotate._wedge import (
    _WEDGE_DEFAULT_FMT,
    _wedge_values_strategy,
)
from publiplots.annotate.bar_custom import _bar_custom_strategy
from publiplots.annotate.bar_values import _bar_values_strategy
from publiplots.annotate.box_custom import _box_custom_strategy
from publiplots.annotate.box_stats import _box_stats_strategy
from publiplots.annotate.point_custom import _point_custom_strategy
from publiplots.annotate.point_values import _point_values_strategy
from publiplots.annotate.violin_custom import _violin_custom_strategy


_STRATEGIES: dict[str, Callable] = {
    "bar_values": _bar_values_strategy,
    "bar_custom": _bar_custom_strategy,
    "box_stats": _box_stats_strategy,
    "box_custom": _box_custom_strategy,
    "point_values": _point_values_strategy,
    "point_custom": _point_custom_strategy,
    "violin_custom": _violin_custom_strategy,
    "wedge_values": _wedge_values_strategy,
}


# Kinds whose rotation may also be named rather than numeric, and the only
# kinds that accept `basis=`. Both are wedge geometry: there is no
# tangent to a bar, and no parent to take a share of.
_WEDGE_KINDS = {"wedge_values"}


# Per-strategy default for `fmt` when the caller did not pass one explicitly.
_DEFAULT_FMT: dict[str, str] = {
    "bar_values": ".2f",
    "bar_custom": "{}",
    "box_stats": ".2f",
    "box_custom": "{}",
    "point_values": ".2f",
    "point_custom": "{}",
    "violin_custom": "{}",
    "wedge_values": _WEDGE_DEFAULT_FMT,
}


def annotate(
    ax: Axes,
    kind: str = "bar_values",
    *,
    fmt: Optional[str] = None,
    anchor: Optional[str] = None,
    offset: float = 1.0,
    color: Union[str, tuple] = "auto",
    pad: Optional[float] = None,
    rotation: Union[float, str] = 0.0,
    labels=None,
    data=None,
    basis=None,
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
    kind : {'bar_values', 'bar_custom', 'box_stats', 'point_values'}, default ``'bar_values'``
        Annotation strategy to use:

        - ``'bar_values'`` — label bars with their values; anchor
          vocabulary ``{"outside", "inside", "base", "center"}``.
        - ``'bar_custom'`` — label bars with user-supplied strings
          from a DataFrame column or a callable
          ``fn(BarRecord) -> str``; anchor vocabulary
          ``{"outside", "inside", "base", "center"}`` (same as
          ``bar_values``).
        - ``'box_stats'`` — label box plots with distributional
          summary statistics.
        - ``'point_values'`` — label scatter or line-plot points;
          anchor vocabulary
          ``{"top", "bottom", "left", "right", "center"}``.
        - ``'wedge_values'`` — label pie and donut wedges with their
          share; anchor vocabulary ``{"inside", "outside", "auto"}``.
          Under ``'auto'`` each label is measured against the room its
          own wedge has and pushed outside, with a leader line, when it
          does not fit. Outside labels make the axis limits grow to
          hold them, and when the chart cannot hold both ``offset`` and
          ``pad``, ``offset`` is honoured exactly and ``pad`` is
          reduced — so on a chart too small for its labels they may end
          up outside the axes rather than at the wrong distance from
          their wedge. Prefer ``anchor='inside'`` or a shorter ``fmt``
          there.
    fmt : str, optional
        Either a bare format spec (e.g. ``".2f"``, ``",.0f"``) or a
        format-string template containing ``{}`` (e.g.
        ``"{:,.1f}%"``). If ``None`` (the default), each strategy picks
        its own default: ``bar_values``, ``box_stats``, and
        ``point_values`` default to ``'.2f'``; ``bar_custom`` defaults
        to ``'{}'`` (raw passthrough of the supplied labels).
        ``wedge_values`` defaults to the wedge's share as a percentage
        to one decimal; passing any spec switches it to formatting the
        wedge's own value instead.
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
    rotation : float or str, default ``0.0``
        Label rotation in degrees, counter-clockwise. For right-angle
        multiples (0, 90, 180, 270) the ``(ha, va)`` alignment is
        auto-remapped so the rotated label still sits at the expected
        offset from the mark, and the categorical axis is also expanded
        so rotated labels do not clip neighbouring marks. Non-right-angle
        rotations are passed through to matplotlib as-is; fine-tuning
        alignment is the caller's responsibility in that case.

        ``wedge_values`` additionally accepts ``'tangential'`` (along
        the wedge's arc) and ``'radial'`` (along its mid-ray), each
        flipped where needed so no label reads upside-down. Those names
        need a mid-ray to be taken from, so any other kind rejects them.
    labels : str or callable, optional
        ``bar_custom`` only. Either a column name to look up in the
        cached ``pp.barplot`` source frame (or in ``data=`` when given),
        or a callable ``fn(BarRecord) -> str`` receiving each bar's
        record and returning the label string. Required when
        ``kind='bar_custom'``; passing it with any other kind is a
        ``TypeError``.
    data : pandas.DataFrame, optional
        ``bar_custom`` only. Overrides the cached source frame for
        column-based labels. Useful when labels live in a different
        DataFrame than the one passed to ``pp.barplot``.
    basis : {'total', 'parent'}, optional
        ``wedge_values`` only: whether a percentage is of the whole
        chart or of the wedge's parent. Identical on a pie, which has
        one parent. Default ``'total'``. Passing it with any other kind
        is a ``TypeError``.
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
    TypeError
        If an option belonging to one strategy is passed with a
        ``kind`` that does not take it: ``basis=`` outside the wedge
        kinds, or ``labels=`` / ``data=`` outside the custom-label
        kinds.

    Notes
    -----
    **One call per plot call.** Each plot function attaches a fresh meta
    cache to ``ax``, and this function consumes the one most recently
    attached. When several plot calls layer onto one axes — the shape x
    color idiom, for instance, one ``pp.pointplot`` per marker shape over a
    shared hue — annotate *inside* the loop::

        for pipeline, marker in [('a', 'o'), ('b', 'D')]:
            pp.pointplot(data=df[df.pipe == pipeline], ..., markers=marker, ax=ax)
            pp.annotate(ax, kind='point_values')     # <- inside

    A single trailing ``pp.annotate(ax, ...)`` would label only the last
    call's marks, because the earlier calls' caches have been replaced.
    Passing ``annotate=`` to each plot call is equivalent and terser.

    The converse also holds: calling this twice after the *same* plot call
    consumes the same cache twice and draws every label twice. See issue
    #235.

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

    Label pie wedges with their share, laid along each arc:

    >>> fig, ax = pp.subplots()
    >>> pp.pieplot(data=df, x='kind', y='n', ax=ax)
    >>> pp.annotate(ax, kind='wedge_values', rotation='tangential')
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
    # Wedge charts add two named orientations, `"tangential"` and
    # `"radial"`, which only mean anything where there is a mid-ray to
    # take them from. Every other kind still requires a finite number.
    if isinstance(rotation, str):
        if kind not in _WEDGE_KINDS:
            raise ValueError(
                f"rotation must be a finite number of degrees for "
                f"kind={kind!r}; the named orientations are wedge-only"
            )
    elif not math.isfinite(rotation):
        raise ValueError("rotation must be a finite number of degrees")

    resolved_fmt = fmt if fmt is not None else _DEFAULT_FMT[kind]

    # labels= and data= are only meaningful for *_custom strategies; reject
    # them for other kinds with a clear error, and forward them conditionally
    # so other strategies' signatures stay unchanged.
    _custom_kinds = {"bar_custom", "box_custom", "point_custom", "violin_custom"}
    extra = {}
    if kind in _custom_kinds:
        extra["labels"] = labels
        extra["data"] = data
    elif labels is not None or data is not None:
        raise TypeError(
            f"labels= and data= are only supported for custom-label kinds "
            f"({sorted(_custom_kinds)}); got kind={kind!r}"
        )

    if basis is not None:
        if kind not in _WEDGE_KINDS:
            raise TypeError(
                f"basis= is only supported for {sorted(_WEDGE_KINDS)}; "
                f"got kind={kind!r}"
            )
        extra["basis"] = basis

    return _STRATEGIES[kind](
        ax, fmt=resolved_fmt, anchor=anchor, offset=offset, color=color, pad=pad,
        rotation=rotation, **extra, **text_kws,
    )
