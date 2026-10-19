"""Public entry point: annotate(ax, kind=..., ...).

Validates inputs and dispatches to a registered strategy. Adding a new
strategy means: write the `_<kind>_strategy` function in its own file and
register it in `_STRATEGIES` below.
"""
from __future__ import annotations

import math
import numbers
from typing import Callable, List, Optional, Union

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._wedge import (
    _LEADER_DEFAULT,
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
from publiplots.annotate.violin_stats import _violin_stats_strategy


_STRATEGIES: dict[str, Callable] = {
    "bar_values": _bar_values_strategy,
    "bar_custom": _bar_custom_strategy,
    "box_stats": _box_stats_strategy,
    "box_custom": _box_custom_strategy,
    "point_values": _point_values_strategy,
    "point_custom": _point_custom_strategy,
    "violin_stats": _violin_stats_strategy,
    "violin_custom": _violin_custom_strategy,
    "wedge_values": _wedge_values_strategy,
}


# Kinds whose rotation may also be named rather than numeric, and the only
# kinds that accept `basis=`, `leader=`, `leader_gap=`, `elbow_length=`,
# `avoid_overlap=` and `label_spacing=`.
# All are wedge geometry: there is no tangent to a bar, no parent to take
# a share of, and no rim for a leader to leave.
_WEDGE_KINDS = {"wedge_values"}

# `offset` for every kind but the wedge ones, which read
# `pp.rcParams["pie.label_offset"]` in their strategy instead.
_DEFAULT_OFFSET = 1.0


# Per-strategy default for `fmt` when the caller did not pass one explicitly.
_DEFAULT_FMT: dict[str, str] = {
    "bar_values": ".2f",
    "bar_custom": "{}",
    "box_stats": ".2f",
    "box_custom": "{}",
    "point_values": ".2f",
    "point_custom": "{}",
    "violin_stats": ".2f",
    "violin_custom": "{}",
    "wedge_values": _WEDGE_DEFAULT_FMT,
}


# Kinds that decide their own default rotation and so receive rotation=None
# when the caller passed none. Every other kind gets 0.0.
_AUTO_ROTATION_KINDS = frozenset({"violin_stats"})


def annotate(
    ax: Axes,
    kind: str = "bar_values",
    *,
    fmt: Optional[str] = None,
    anchor: Optional[str] = None,
    offset: Optional[float] = None,
    color: Union[str, tuple] = "auto",
    pad: Optional[float] = None,
    rotation: Union[float, str, None] = None,
    labels=None,
    data=None,
    basis=None,
    leader=_LEADER_DEFAULT,
    leader_gap: Optional[float] = None,
    elbow_length: Optional[float] = None,
    avoid_overlap: Optional[bool] = None,
    label_spacing: Optional[float] = None,
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
    kind : {'bar_values', 'bar_custom', 'box_stats', 'box_custom', 'violin_stats', 'violin_custom', 'point_values', 'point_custom', 'wedge_values'}, default ``'bar_values'``
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
        - ``'violin_stats'`` — the same statistics for violin plots, with
          default anchor ``"inside"``: on the forward side of the median
          line, inside the body. Labels on dodged vertical violins turn
          90° to fit their body unless ``rotation`` is given.
        - ``'point_values'`` — label scatter or line-plot points;
          anchor vocabulary
          ``{"top", "bottom", "left", "right", "center"}``.
        - ``'wedge_values'`` — label pie and donut wedges with their
          share; anchor vocabulary ``{"inside", "outside", "auto"}``.
          Under ``'auto'`` each label is measured against the room its
          own wedge has and pushed outside when it does not fit. An
          outside label hangs off its wedge's mid-ray in the ``leader``
          style: ``'radial'`` (the default) runs a line ``offset`` mm
          straight out from the rim and centres the label on the mid-ray
          ``leader_gap`` mm beyond its end; ``'elbow'`` bends that line
          into a horizontal tail ``elbow_length`` mm long, with the label
          beside it; ``None`` draws no line and centres the label so its
          nearest point is ``offset`` mm from the rim. Outside labels
          are decorations, like a title: the axis limits are left alone,
          so the disc is the same size whichever anchor is used, and the
          labels extend beyond the axes. On an axes from
          :func:`publiplots.subplots` the layout grows the figure to
          hold them; on any other axes nothing does, and labels that
          reach past the figure edge are cut off. Crowded outside labels
          are moved apart (``avoid_overlap``). ``pad`` has no effect for
          this kind, and is not validated for it.
    fmt : str, optional
        Either a bare format spec (e.g. ``".2f"``, ``",.0f"``) or a
        format-string template containing ``{}`` (e.g.
        ``"{:,.1f}%"``). If ``None`` (the default), each strategy picks
        its own default: ``bar_values``, ``box_stats``,
        ``point_values`` and ``violin_stats`` default to ``'.2f'``; the
        ``*_custom`` kinds default to ``'{}'`` (raw passthrough of the
        supplied labels). ``wedge_values`` defaults to the wedge's share
        as a percentage to one decimal; passing any spec switches it to
        formatting the wedge's own value instead.
    anchor : str, optional
        Where to place the label relative to the mark.
        Strategy-specific — see ``kind`` above. If ``None``, each
        strategy picks its own default.
    offset : float, optional
        Gap in millimetres between the mark (or errorbar cap, when
        present) and the label, applied outward in the anchor
        direction. ``1.0`` when not given, except for ``wedge_values``,
        where it is the distance from the rim to the leader's end (to
        the nearest point of the label, in a straight line, with
        ``leader=None``) and defaults to
        ``pp.rcParams["pie.label_offset"]`` (``2.0``).
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

        Ignored by ``wedge_values``, whose outside labels sit beyond
        the axes edge rather than inside it.
    rotation : float or str, optional
        Label rotation in degrees, counter-clockwise. Defaults to ``0``,
        except ``kind='violin_stats'``, which turns labels on dodged
        vertical violins 90°. For right-angle
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
    leader : {'radial', 'elbow', None}, optional
        ``wedge_values`` only: how an outside label is joined to its
        wedge. ``'radial'`` is a straight line along the mid-ray, with
        the label centred on that ray; ``'elbow'`` adds a horizontal
        tail toward the label's side (right for a mid-ray pointing
        straight up or down), with the label left- or right-aligned
        beside it; ``None`` or ``False`` draws no line. Defaults to
        ``pp.rcParams["pie.leader"]`` (``'radial'``). With
        ``avoid_overlap`` on, a pie whose outside labels the bounded
        circle pass does not keep apart draws ``'elbow'`` labels, with
        their lines, whichever of ``'radial'`` or ``None`` was asked for
        (see ``avoid_overlap``). A rotated label
        is placed by its rotated bounding box in every style. Passing it
        with any other kind is a ``TypeError``.
    leader_gap : float, optional
        ``wedge_values`` only: millimetres from the leader's end to the
        label -- the straight-line distance to the nearest point of the
        label's box, so it reads the same at every angle. For
        ``'elbow'`` that is horizontal, from the end of the tail.
        Defaults to ``pp.rcParams["pie.leader_gap"]`` (``0.8``); unused
        with ``leader=None``. Passing it with any other kind is a
        ``TypeError``. For ``'radial'``, and for ``offset`` with
        ``leader=None``, the label's box is measured with the renderer
        at ``figure.dpi``, so the gap is exact there; text metrics at
        another resolution differ slightly, and a figure saved at
        ``savefig.dpi`` 600 can show the gap up to about 0.05 mm off.
    elbow_length : float, optional
        ``wedge_values`` only: millimetres of an ``'elbow'`` leader's
        horizontal tail. Defaults to ``pp.rcParams["pie.elbow_length"]``
        (``2.0``); unused by the other styles. Passing it with any other
        kind is a ``TypeError``.
    avoid_overlap : bool, optional
        ``wedge_values`` only: move outside labels that would overlap
        apart, by as little as keeps every pair ``label_spacing`` apart.
        ``'elbow'`` labels move up or down their column, the elbow
        moving with them so the tail stays level, and a moved label's
        leader is kept from running through another label (a leader
        that already did so unmoved, as a rotated label's can, is not
        always cleared). ``'radial'`` and unled labels
        move round the circle onto a ray off their wedge's mid-ray, a
        radial leader leaning to follow -- but never further than
        ``acos(R / (R + offset))`` from the mid-ray, ``R`` the wedge's
        radius, the furthest a leader can lean without cutting back
        across the chart (about 25 degrees at 40mm and the default
        ``offset``). Their order is kept, a label that crowds nothing
        does not move, inside labels never move, and no leader enters
        the disc. **When the bounded solve does not clear every
        conflict** -- a run of slivers side by side, or more labels than
        fit round the circle, and also some pies where a tight radial
        packing exists but only with leaders lying close to the rim's
        tangent, which this solve does not reach for -- every outside
        label of the axes is laid out as an ``'elbow'`` label instead,
        leader and all (drawn too when ``leader=None``), using the
        ``offset``, ``leader_gap`` and ``elbow_length`` in effect, and
        spread up and down its column, which always has room. How many
        slivers that takes depends on where they sit and on the chart's
        size: on a 25 to 40mm pie, four to six 0.5% wedges already do
        near twelve or six o'clock, where labels stand side by side, and
        nine to eleven near three or nine, where they stack; a larger pie
        takes more. No warning is issued; pass ``min_share`` to
        :func:`publiplots.pieplot` to pool the slivers if the radial
        style matters more, or ``avoid_overlap=False`` to keep every
        label on its mid-ray whatever overlaps. A pie the circle pass
        clears keeps the style asked for. Defaults to ``pp.rcParams["pie.avoid_overlap"]`` (``True``);
        ``False`` places every label on its own mid-ray. Anything but a
        bool is a ``TypeError``, as is passing it with any other kind.
    label_spacing : float, optional
        ``wedge_values`` only: the least gap in millimetres kept between
        two outside labels' boxes -- the straight-line distance between
        their nearest points -- and between a moved elbow label and the
        disc. Defaults to ``pp.rcParams["pie.label_spacing"]``
        (``0.4``); unused with ``avoid_overlap=False``. Validated as
        ``leader_gap`` is, and a ``TypeError`` with any other kind.
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
        ``offset`` is negative, or ``pad`` is negative for any kind but
        ``wedge_values``.
    TypeError
        If an option belonging to one strategy is passed with a
        ``kind`` that does not take it: ``basis=``, ``leader=``,
        ``leader_gap=``, ``elbow_length=``, ``avoid_overlap=`` or
        ``label_spacing=`` outside the wedge kinds, or
        ``labels=`` / ``data=`` outside the custom-label kinds.

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

    Put every wedge label outside, on elbow leaders:

    >>> pp.annotate(ax, kind='wedge_values', anchor='outside',
    ...             leader='elbow')
    """
    if kind not in _STRATEGIES:
        raise ValueError(
            f"unknown kind={kind!r}; known: {sorted(_STRATEGIES)}"
        )
    # The wedge kinds resolve an omitted `offset` from their own rcParam,
    # so it reaches them as None. Every other kind keeps its 1.0, and
    # `pad` still falls back to the offset actually in effect.
    if offset is None and kind not in _WEDGE_KINDS:
        offset = _DEFAULT_OFFSET
    # The wedge kinds validate `offset` in their strategy, with their
    # other lengths and after the rcParam default is read, so every
    # wedge length reports a bad value the same way.
    if offset is not None and kind not in _WEDGE_KINDS:
        if not isinstance(offset, numbers.Real):
            raise TypeError(
                f"offset must be a number of millimetres; got "
                f"{type(offset).__name__} {offset!r}"
            )
        if offset < 0:
            raise ValueError("offset must be >= 0")
    if pad is None:
        pad = offset if offset is not None else _DEFAULT_OFFSET
    # `pad` has no effect for the wedge kinds, so a value there is not
    # validated: refusing an argument that changes nothing would only make
    # a call shared across kinds fail on the pie.
    if kind not in _WEDGE_KINDS and pad < 0:
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
    elif rotation is not None and not math.isfinite(rotation):
        raise ValueError("rotation must be a finite number of degrees")
    if rotation is None and kind not in _AUTO_ROTATION_KINDS:
        rotation = 0.0

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

    leader_options = {
        name: value
        for name, value, unset in (("leader", leader, _LEADER_DEFAULT),
                                   ("leader_gap", leader_gap, None),
                                   ("elbow_length", elbow_length, None),
                                   ("avoid_overlap", avoid_overlap, None),
                                   ("label_spacing", label_spacing, None))
        if value is not unset
    }
    if leader_options:
        if kind not in _WEDGE_KINDS:
            names = ", ".join(f"{name}=" for name in leader_options)
            verb = "is" if len(leader_options) == 1 else "are"
            raise TypeError(
                f"{names} {verb} only supported for {sorted(_WEDGE_KINDS)}; "
                f"got kind={kind!r}"
            )
        extra.update(leader_options)

    return _STRATEGIES[kind](
        ax, fmt=resolved_fmt, anchor=anchor, offset=offset, color=color, pad=pad,
        rotation=rotation, **extra, **text_kws,
    )
