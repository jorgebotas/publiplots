"""
Bar plot functions for publiplots.

This module provides publication-ready bar plot visualizations with
flexible styling and grouping options.
"""

import warnings
from typing import Literal, Optional, List, Dict, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd

from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.hatches import resolve_hatch_map
from publiplots.utils import is_categorical, as_categorical, create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    resolve_legend_flags,
)
from publiplots.utils.plot_legend import render_entries
from publiplots.utils.rounding import apply_border_radius, normalize_border_radius
from publiplots.utils.transparency import ArtistTracker
from publiplots.annotate._builders import (
    build_from_barplot_call,
    build_from_stacked_barplot_call,
)
from publiplots.annotate._splits import BarSplitSpec, _categories_in_draw_order

_SPLIT_SEPARATOR = "---"

def barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    hatch: Optional[str] = None,
    color: Optional[str] = None,
    edgecolor: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    linewidth: Optional[float] = None,
    capsize: Optional[float] = None,
    alpha: Optional[float] = None,
    border_radius: Optional[Union[float, Tuple[float, float]]] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    hatch_map: Optional[Dict[str, str]] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    annotate: Union[bool, Dict, None] = None,
    errorbar: str = "se",
    gap: float = 0.1,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    hatch_order: Optional[List[str]] = None,
    multiple: Literal["dodge", "stack", "fill", "gain"] = "dodge",
    stack_by: Optional[Literal["hue", "hatch"]] = None,
    **kwargs
) -> Axes:
    """
    Create a publication-ready bar plot.

    Aggregates long-form data by a categorical axis (``x`` or ``y``) and
    draws one bar per group, with optional color / hatch grouping and
    optional stacking. Under the hood ``multiple="dodge"`` (the default)
    wraps :func:`seaborn.barplot` and applies publiplots' palette, hatch,
    transparency, and legend-stash styling in a post-draw pass;
    ``multiple="stack"`` / ``"fill"`` bypasses seaborn (which has no
    stacked mode) and draws segments directly with ``ax.bar`` /
    ``ax.barh`` at integer category positions with cumulative
    ``bottom=`` / ``left=``.

    Parameters
    ----------
    data : DataFrame
        Input data in long form — one row per observation.
    x, y : str
        Column names for the two axes. Exactly one must be categorical
        (category dtype, object, or string) and the other numeric.
        ``x`` categorical + ``y`` numeric → vertical bars;
        ``y`` categorical + ``x`` numeric → horizontal bars. If neither
        column is categorical, raises :class:`ValueError`.
    hue : str, optional
        Column name for color grouping. When distinct from the
        categorical axis, bars split within each category — side-by-side
        under ``multiple="dodge"``, stacked under ``multiple="stack"`` /
        ``"fill"``. When equal to the categorical axis, hue collapses to
        per-category coloring (no splitting).
    hatch : str, optional
        Column name driving a **second categorical dimension** encoded
        via hatch textures — the texture-based analogue of ``hue``.
        Under ``multiple="dodge"``: when ``hatch`` is distinct from both
        ``hue`` and the categorical axis, bars split along a second
        dodge dimension and the legend renders two entries (see
        ``hue``-and-hatch double-split in the bar gallery). Under
        ``multiple="stack"`` / ``"fill"``: ``hatch`` can drive the stack
        on its own (leave ``hue`` unset) — useful for B&W-friendly
        stacks — or share the column with ``hue`` (``hatch=hue``) to
        pattern each stack segment by its hue level. When both ``hue``
        and ``hatch`` are distinct non-categorical columns, pass
        ``stack_by="hue"`` or ``stack_by="hatch"`` to pick which
        dimension stacks; the other is dodged side-by-side within each
        category.

        Patterns are drawn from :data:`publiplots.HATCH_PATTERNS` by
        default; density is controlled globally via
        :func:`publiplots.set_hatch_mode` (``"dense"``, ``"sparse"``, or
        ``"off"``) or overridden per-plot with ``hatch_map``. Call
        :func:`publiplots.list_hatch_patterns` to see the catalog.
    color : str, optional
        Fixed color for all bars when no hue split is active. Accepts
        any matplotlib color string (``"#ff0000"``, ``"red"``, etc.).
        Falls back to ``rcParams["color"]``.
    edgecolor : str, optional
        Explicit edge color override. When ``None`` (the default, also
        the default of ``rcParams["edgecolor"]``), the edge matches the
        face color — publiplots' double-layer transparent-fill styling.
        Set to a specific color (``"black"``) to draw bars with
        consistent outlines regardless of fill.
    ax : Axes, optional
        Matplotlib axes to draw on. If ``None``, creates a new figure
        via :func:`publiplots.layout.subplots` so the publiplots layout
        manager can keep geometry stable. To control axes dimensions,
        pre-create with ``pp.subplots(axes_size=(w_mm, h_mm))`` and
        pass ``ax=``.
    title : str or None, default=""
        Plot title. Pass ``None`` to leave the existing title unchanged.
    xlabel, ylabel : str or None, default=""
        Axis labels. Pass ``None`` to leave the existing label unchanged.
    linewidth : float, optional
        Width of bar edges and hatch strokes. Defaults to
        ``rcParams["lines.linewidth"]``.
    capsize : float, optional
        Width of errorbar caps in data units (dodge path only — stacked
        bars drop errorbars). Defaults to ``rcParams["capsize"]``.
    alpha : float, optional
        Transparency of the bar *fill* (edges stay fully opaque).
        ``0`` = outlined bars only; ``1`` = solid fill. Defaults to
        ``rcParams["alpha"]``.
    border_radius : float or (top_mm, bottom_mm) tuple, optional
        Corner radius for the bars, in **millimeters** (print-consistent,
        independent of data-axis range). A scalar rounds all four corners
        symmetrically; a 2-tuple rounds the top and bottom independently,
        enabling infographic-style bars with rounded tops and flat
        baselines (``border_radius=(1.5, 0)``). Defaults to the
        ``bar.border_radius`` rcParam (``(0, 0)`` = flat). Set globally
        via ``pp.rcParams['bar.border_radius'] = 1.5``.
    palette : str, dict, or list, optional
        Color palette. Can be:

        - ``str``: seaborn palette name or publiplots palette name
        - ``dict``: explicit ``{hue_level: color}`` mapping
        - ``list``: list of colors assigned in hue-level order

        Only used when a hue split is active.
    hatch_map : dict, optional
        Mapping from hatch-column values to matplotlib hatch-pattern strings,
        overriding the per-``hatch_mode`` defaults. Matplotlib hatches
        interpret: ``/`` ``\\`` ``|`` ``-`` ``+`` ``x`` ``o`` ``O`` ``.`` ``*``;
        repeating a glyph increases density (e.g. ``"///"``).

        Example (two-level hatch with a plain control group)::

            hatch_map={"control": "", "treated": "///"}

        :data:`publiplots.HATCH_PATTERNS` lists the built-in patterns the
        ``hatch_mode`` resolver picks from; call
        :func:`publiplots.list_hatch_patterns` to print them.
    legend : bool or dict, default=True
        Whether to stash & render the legend. Accepts ``bool`` or
        ``dict[kind, bool]`` for per-kind control (e.g.
        ``legend={"hatch": False}`` hides just the hatch entry in a
        double-split bar plot). ``False`` suppresses stashing entirely,
        so a figure-level :func:`publiplots.legend` group won't claim
        any entries from this axes.
    legend_kws : dict, optional
        Extra kwargs forwarded to the legend builder. Notable keys:
        ``inside=True`` with ``loc=...`` places the legend inside the
        axes using matplotlib's corner-based placement; ``hue_label`` /
        ``hatch_label`` override the title shown above each entry.
    annotate : bool or dict, optional
        If truthy, label each bar with its aggregated value. Pass a
        ``dict`` to forward options to :func:`publiplots.annotate`:
        ``fmt`` (Python format spec), ``anchor``
        (``"outside"``/``"inside"``/``"top"``/``"left"``/``"right"``),
        ``offset`` (mm), ``color``, ``rotation`` (degrees).

        Under ``multiple="stack"`` / ``"fill"`` the default anchor is
        ``"inside"`` and one label is produced per drawn segment; under
        ``multiple="dodge"`` the default anchor is ``"outside"`` and
        one label is produced per bar.
    errorbar : str or None, default="se"
        Error bar type for the dodge path: ``"se"`` (standard error),
        ``"sd"`` (standard deviation), ``"ci"`` (confidence interval),
        tuple ``("pi", q)`` for percentile intervals, callable, or
        ``None`` for no error bars. Under ``multiple="stack"`` /
        ``"fill"`` errorbars are always dropped with a
        :class:`UserWarning` — per-segment errors are not additive
        without covariance info.
    gap : float, default=0.1
        Gap between adjacent bars, as a fraction of bar width.
        Under ``"dodge"`` this is the gap between hue/hatch levels
        within a category; under ``"stack"`` / ``"fill"`` this is the
        gap between stacks across the categorical axis.
    order : list, optional
        Order of the categorical axis. Determines both draw order and
        the tick-label sequence.
    hue_order : list, optional
        Explicit order of hue levels. Under ``"stack"`` / ``"fill"``,
        also determines stack order bottom-to-top (first level at the
        base, last level on top) and the legend entry order.
    hatch_order : list, optional
        Order of hatch levels. Under ``"stack"`` / ``"fill"`` with
        ``hatch`` driving the stack, determines stack order
        bottom-to-top.
    multiple : {"dodge", "stack", "fill"}, default="dodge"
        How to arrange bars across the secondary (hue / hatch)
        categorical dimension within each category of the primary axis.

        - ``"dodge"`` (default): levels sit **side-by-side** within each
          category via seaborn's dodging. Current behavior, unchanged —
          supports errorbars, ``hue + hatch`` double-split, and the full
          four-case legend dispatcher.
        - ``"stack"``: levels sit **on top of each other** within each
          category, each segment's base set to the cumulative height
          (or width, horizontal) of the levels below. Drawn directly
          with ``ax.bar`` / ``ax.barh`` — seaborn's barplot has no stack
          mode.
        - ``"fill"``: stack and then normalize so every stack sums to
          1.0 (100%-stacked bars — good for showing proportions whose
          totals differ across categories).

        Stacking requires at least one of ``hue`` / ``hatch`` to be set
        and distinct from the categorical axis. If *neither* is set,
        raises :class:`ValueError`. When *both* are distinct
        non-categorical columns, use ``stack_by=`` (below) to pick the
        stack dimension — the other is dodged side-by-side within each
        category. ``hue == hatch`` (both drive a single stack, patterns
        overlaid on colored swatches) is allowed.
    stack_by : {"hue", "hatch"}, optional
        Required when ``multiple in {"stack", "fill"}`` *and* both
        ``hue`` and ``hatch`` are distinct non-categorical columns.
        Picks which dimension accumulates along the value axis; the
        other dimension dodges side-by-side within each category (so
        each category shows ``N_other`` dodged stacks, each of
        ``N_stack_by`` segments). Ignored when only one of
        ``hue`` / ``hatch`` is active. Pass ``None`` in the single-dim
        case — the sole split dimension becomes the stack.
    **kwargs
        Additional keyword arguments passed to :func:`seaborn.barplot`
        in the dodge path. Ignored in the stack/fill path. ``figsize``
        is always rejected — use ``pp.subplots(axes_size=...)`` and
        pass ``ax=``.

    Returns
    -------
    Axes
        The axes where the plot was drawn. Recover the figure handle
        with ``ax.get_figure()``.

    Raises
    ------
    ValueError
        If neither ``x`` nor ``y`` is categorical; if ``multiple`` is
        not one of ``"dodge"``, ``"stack"``, ``"fill"``; if
        ``multiple="stack"|"fill"`` is requested without a stack column
        (``hue`` / ``hatch``); if ``multiple="stack"|"fill"`` is
        requested with both ``hue`` and ``hatch`` set as distinct
        non-categorical columns but no ``stack_by=`` was provided.
    TypeError
        If ``figsize`` is passed (publiplots owns figure geometry via
        :func:`publiplots.subplots`).

    Warns
    -----
    UserWarning
        If ``errorbar`` is set under ``multiple="stack"|"fill"``; the
        errorbar is silently dropped (per-segment errors are not
        additive without covariance information).

    Notes
    -----
    **Performance.** The dodge path delegates aggregation to seaborn;
    the stack/fill path aggregates once in a single pass via the shared
    :class:`publiplots.annotate._splits.BarSplitSpec` iterator (the same
    deterministic draw order used for annotate pairing).

    **Legend stash.** Every call stashes :class:`LegendEntry` objects on
    the axes (unless ``legend=False``); per-axes legends render unless
    a figure-level :func:`publiplots.legend` group claims them.

    Examples
    --------
    Simple bar plot:

    >>> ax = pp.barplot(data=df, x="category", y="value")

    Color grouping via ``hue``:

    >>> ax = pp.barplot(data=df, x="time", y="value",
    ...                 hue="group", palette="pastel")

    Two categorical dimensions via ``hue`` + ``hatch`` (side-by-side):

    >>> ax = pp.barplot(
    ...     data=df, x="cell_type", y="viability",
    ...     hue="treatment", hatch="time",
    ...     palette={"Vehicle": "#8E8EC1", "Drug": "#60a8a8"},
    ...     hatch_map={"24h": "", "48h": "///"},
    ... )

    Value labels on each bar:

    >>> ax = pp.barplot(data=df, x="category", y="value",
    ...                 annotate={"fmt": ".2f"})

    Stacked bars (``multiple="stack"``) with per-segment labels:

    >>> ax = pp.barplot(
    ...     data=df, x="cohort", y="count", hue="stage",
    ...     multiple="stack", errorbar=None,
    ...     hue_order=["Early", "Mid", "Late"],
    ...     annotate={"fmt": ".0f"},
    ... )

    100%-stacked proportions with percentage labels:

    >>> ax = pp.barplot(
    ...     data=df, x="cohort", y="count", hue="stage",
    ...     multiple="fill", errorbar=None,
    ...     annotate={"fmt": ".0%"},
    ... )

    B&W-friendly stack keyed by hatch (no hue):

    >>> ax = pp.barplot(
    ...     data=df, x="cohort", y="count", hatch="stage",
    ...     multiple="stack", errorbar=None, color="#5D83C3",
    ...     hatch_map={"Early": "", "Mid": "//", "Late": "xx"},
    ... )

    Horizontal stacked bars:

    >>> ax = pp.barplot(
    ...     data=df, x="count", y="cohort", hue="stage",
    ...     multiple="stack", errorbar=None,
    ... )

    Two categoricals: stack one, dodge the other (``stack_by``):

    >>> ax = pp.barplot(
    ...     data=df, x="cell_type", y="viability",
    ...     hue="treatment", hatch="time",
    ...     multiple="stack", stack_by="hue", errorbar=None,
    ...     palette={"Vehicle": "#8E8EC1", "Drug": "#60a8a8"},
    ...     hatch_map={"24h": "", "48h": "///"},
    ... )

    See Also
    --------
    publiplots.histplot : Histogram plot (also supports
        ``multiple="stack"``/``"fill"`` via seaborn).
    publiplots.set_hatch_mode : Set the global hatch-density mode.
    publiplots.list_hatch_patterns : Print the built-in hatch patterns.
    publiplots.annotate : Add value labels to bars (see ``annotate=``
        above for the subset promoted to ``pp.barplot``).
    publiplots.legend : Figure-level legend group that claims stashed
        entries across multiple axes.
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    capsize = resolve_param("capsize", capsize)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Create figure via pp.subplots to install SubplotsAutoLayout; users who
    # want custom dimensions should compose with pp.subplots(axes_size=...)
    # before calling and pass ax=.
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Find out categorical axis
    categorical_axis = x if is_categorical(data[x]) else y
    if not (is_categorical(data[x]) or is_categorical(data[y])):
        raise ValueError(
            "At least one of x or y must be categorical. "
            "Run data[x].astype('category') or data[y].astype('category')"
        )

    # Ensure category dtype on every column we'll touch with the .cat accessor
    data = data.copy()
    data[categorical_axis] = as_categorical(data[categorical_axis])
    if hue is not None and hue != categorical_axis:
        data[hue] = as_categorical(data[hue])
    if hatch is not None and hatch != categorical_axis:
        data[hatch] = as_categorical(data[hatch])

    # Get hue palette and hatch mappings
    palette = resolve_palette_map(
        values=data[hue].unique() if hue is not None else None,
        palette=palette,
    )
    hatch_map = resolve_hatch_map(
        values=data[hatch].unique() if hatch is not None else None,
        hatch_map=hatch_map,
    )

    # Honor hue_order / hatch_order in the resolved dict key order — the
    # legend dispatcher iterates these dicts in insertion order, so an
    # unordered dict ignores the user's request. Missing levels pass
    # through unchanged; extra levels (beyond the dict keys) are tolerated
    # but warn-worthy behavior we defer to resolve_palette_map.
    if hue_order is not None and palette:
        palette = {k: palette[k] for k in hue_order if k in palette}
    if hatch_order is not None and hatch_map:
        hatch_map = {k: hatch_map[k] for k in hatch_order if k in hatch_map}


    prepareA = hue is not None and (hue != categorical_axis)
    prepareB = hatch is not None and (hatch != categorical_axis)
    data = _prepare_split_data(
        data,
        hue,
        hatch,
        categorical_axis,
        orderA=(hue_order or list(palette.keys())) if prepareA else None,
        orderB=(hatch_order or list(hatch_map.keys())) if prepareB else None,
        order_categorical_axis=order,
    )

    # Resolve which dimensions actually dodge via the shared spec (single
    # source of truth — annotate builders use the same rules).
    _split = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=hatch, categorical_axis=categorical_axis,
    )
    split_by_hatch = _split.split_hatch is not None
    double_split = _split.split_hue is not None and _split.split_hatch is not None

    if multiple not in ("dodge", "stack", "fill", "gain"):
        raise ValueError(
            "barplot: multiple must be one of 'dodge', 'stack', 'fill', "
            f"'gain'; got {multiple!r}"
        )

    if multiple in ("stack", "fill", "gain"):
        tracker = ArtistTracker(ax)
        _draw_stacked(
            ax=ax, data=data, x=x, y=y, hue=hue, hatch=hatch,
            categorical_axis=categorical_axis, split=_split,
            multiple=multiple, stack_by=stack_by,
            palette=palette, hatch_map=hatch_map,
            color=color, edgecolor=edgecolor, linewidth=linewidth,
            errorbar=errorbar, gap=gap,
        )
        radius = normalize_border_radius(
            resolve_param("bar.border_radius", border_radius)
        )
        apply_border_radius(tracker.get_new_patches(), radius, ax)
        tracker.apply_transparency(on="patches", face_alpha=alpha, edge_alpha=1.0)
        # Reuse the dodge path's four-case legend dispatcher — the legend
        # treatment is identical whether bars are dodged or stacked.
        _legend(
            ax=ax,
            hue=hue, hatch=hatch, categorical_axis=categorical_axis,
            alpha=alpha, linewidth=linewidth,
            color=color, edgecolor=edgecolor,
            palette=palette, hatch_map=hatch_map,
            kwargs=legend_kws, legend=legend,
        )
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if title is not None: ax.set_title(title)
        if annotate:
            ax._publiplots_bar_meta = build_from_stacked_barplot_call(
                ax=ax, data=data, x=x, y=y, hue=hue, hatch=hatch,
                categorical_axis=categorical_axis, palette=palette,
            )
            from publiplots.annotate import annotate as _annotate_fn
            opts = dict(annotate) if isinstance(annotate, dict) else {}
            opts.setdefault("anchor", "inside")
            _annotate_fn(ax, kind="bar_values", **opts)
        return ax

    sns_hue = hue
    sns_palette = palette

    if _split.split_hue is None:
        # hue omitted or equals categorical axis — don't pass it to seaborn.
        sns_hue = None
        sns_palette = None

    if split_by_hatch:
        if double_split:
            sns_hue = f"{hue}_{hatch}"
            sns_palette = {
                x: palette[x.split(_SPLIT_SEPARATOR)[0]]
                for x in data[f"{hue}_{hatch}"].cat.categories
            }
        else:
            sns_hue = hatch
            sns_palette = None

    # Prepare kwargs for seaborn barplot
    barplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": sns_hue,
        "color": color if sns_hue is None else None,
        "palette": sns_palette if sns_hue else None,
        "fill": False,
        "linewidth": linewidth,
        "capsize": capsize,
        "ax": ax,
        "err_kws": {"linewidth": linewidth},
        "errorbar": errorbar,
        "gap": gap,
        "legend": False,
    }

    # Merge with user-provided kwargs
    barplot_kwargs.update(kwargs)

    # Create bars with fill and edges
    barplot_kwargs["fill"] = False

    # Track artists before plotting
    tracker = ArtistTracker(ax)

    # Create bars
    sns.barplot(**barplot_kwargs)

    # Paint face, hatch, and edge in one deterministic pass. Replaces the
    # prior chain of ``seaborn fill=False puts palette on edge`` + ``copy
    # edge to face later`` tricks which were order-dependent and broke
    # under mpl 3.10.8 when ``color=`` was set alongside ``hatch=`` (#105).
    _paint_bars(
        patches=tracker.get_new_patches(),
        errorbars=tracker.get_new_lines(),
        data=data,
        split=_split,
        hue=hue,
        hatch=hatch,
        categorical_axis=categorical_axis,
        linewidth=linewidth,
        color=color,
        edgecolor=edgecolor,
        palette=palette,
        hatch_map=hatch_map,
    )

    # Round bar corners per rcParam / kwarg (no-op on (0, 0)). Runs AFTER
    # _paint_bars so face/edge/hatch are already on each Rectangle and get
    # copied over by apply_border_radius; runs BEFORE apply_transparency so
    # transparency applies to the new FancyBboxPatches via tracker snapshot
    # diff (swapped-in patches are not in the pre-plot snapshot and are
    # picked up by get_new_patches).
    radius = normalize_border_radius(
        resolve_param("bar.border_radius", border_radius)
    )
    apply_border_radius(tracker.get_new_patches(), radius, ax)

    # Apply differential transparency to face vs edge
    tracker.apply_transparency(on="patches", face_alpha=alpha, edge_alpha=1.0)

    # Stash entries and render per-axis unless claimed by a figure-level group.
    _legend(
        ax=ax,
        hue=hue,
        hatch=hatch,
        categorical_axis=categorical_axis,
        alpha=alpha,
        linewidth=linewidth,
        color=color,
        edgecolor=edgecolor,
        palette=palette,
        hatch_map=hatch_map,
        kwargs=legend_kws,
        legend=legend,
    )

    # Set labels
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    if annotate:
        ax._publiplots_bar_meta = build_from_barplot_call(
            ax=ax, data=data, x=x, y=y, hue=hue, hatch=hatch,
            categorical_axis=categorical_axis,
            palette=palette, errorbar=errorbar,
        )
        from publiplots.annotate import annotate as _annotate_fn
        opts = annotate if isinstance(annotate, dict) else {}
        _annotate_fn(ax, kind="bar_values", **opts)

    return ax


# =============================================================================
# Helper Functions
# =============================================================================


def _prepare_split_data(
        data: pd.DataFrame, 
        colA: str,
        colB: str,
        categorical_axis: str,
        orderA: Optional[List[str]] = None,
        orderB: Optional[List[str]] = None,
        order_categorical_axis: Optional[List[str]] = None,
    ) -> pd.DataFrame:
    """
    Prepare data for split bar plotting by creating a combined column.
    
    Parameters
    ----------
    data : DataFrame
        Input data
    colA : str
        Column name for first split
    colB : str
        Column name for second split
    
    orderA : list, optional
        Order of column A values. If provided, data will be sorted to match this order.
    orderB : list, optional
        Order of column B values. If provided, data will be sorted to match this order.
    
    Returns
    -------
    DataFrame
        Data with new combined column for proper bar separation and sorted by the order of the columns
        New column name is f"{colA}_{colB}"
    """
    data = data.copy()

    # If order is provided, ensure the split column follows that order
    prepareA = colA is not None and orderA is not None
    prepareB = colB is not None and orderB is not None
    if prepareA:
        data[colA] = as_categorical(data[colA], categories=orderA).cat.remove_unused_categories()
        data = data.sort_values([colA])
    if prepareB:
        data[colB] = as_categorical(data[colB], categories=orderB).cat.remove_unused_categories()
        data = data.sort_values([colB])

    # Sort the data by the columns in the order of the columns
    columns = ([colA] if prepareA else []) + ([colB] if prepareB else [])
    if order_categorical_axis is not None:
        data[categorical_axis] = as_categorical(
            data[categorical_axis], categories=order_categorical_axis
        ).cat.remove_unused_categories()
        columns.insert(0, categorical_axis)
    data.sort_values(columns, inplace=True)

    if prepareA and prepareB:
        # Create a combined column that seaborn will use to separate bars
        data[f"{colA}_{colB}"] = as_categorical(
            data[colA].astype(str) + _SPLIT_SEPARATOR + data[colB].astype(str)
        )
    return data

def _paint_bars(
        patches: List,
        errorbars: List,
        data: pd.DataFrame,
        split,
        hue: Optional[str],
        hatch: Optional[str],
        categorical_axis: str,
        linewidth: float,
        color: Optional[str],
        edgecolor: Optional[str],
        palette: Optional[Dict[str, str]],
        hatch_map: Optional[Dict[str, str]],
    ) -> None:
    """Paint face, hatch, and edge on each bar in a single deterministic pass.

    For every new patch produced by ``sns.barplot(fill=False)``:

    1. Choose the face color from the appropriate source (``palette``
       keyed by axis/hue/hatch category, or scalar ``color``).
    2. Apply the hatch pattern (empty when ``hatch`` is None).
    3. Set the edge color to ``edgecolor`` if provided, else to the face
       color so bars read as solid shapes.
    4. Match the paired error-bar line color.

    Iteration order is taken from ``BarSplitSpec.iter_draw_order`` — the
    same authoritative iterator the annotate layer uses — so the
    per-bar ``(cat, hue_value, hatch_value)`` triples line up 1:1 with
    the patches seaborn drew.

    Single-pass ordering avoids the cross-version matplotlib fragility
    where reading ``patch.get_edgecolor()`` after an in-place mutation
    returned a stale sentinel and black face colors leaked through
    (issue #105).
    """
    bar_patches = [p for p in patches if hasattr(p, "get_height")]
    triples = list(split.iter_draw_order(data))

    for idx, patch in enumerate(bar_patches):
        if idx >= len(triples):
            # Safety: seaborn drew more patches than the spec iterates.
            # Shouldn't happen; skip rather than crash.
            continue
        cat, hue_value, hatch_value = triples[idx]

        # Resolve face color. The branches are mutually exclusive and
        # map 1:1 to the legend dispatch in ``_legend``; keep them in
        # sync.
        if split.split_hue is not None:
            # ``hue`` genuinely dodges. ``hue_value`` is authoritative.
            face_color = palette[hue_value]
        elif hue is not None and hue == categorical_axis and palette:
            # hue collapses onto the categorical axis — color by cat.
            face_color = palette.get(cat, color)
        elif hue is not None and hue == hatch and palette:
            # hue == hatch: palette keyed by hatch value.
            face_color = palette.get(hatch_value, color)
        else:
            # No hue split — fall back to the scalar ``color``.
            face_color = color

        patch.set_facecolor(face_color)

        if hatch is not None:
            # Resolve the hatch key:
            # - split_hatch active → hatch_value is authoritative.
            # - hue == hatch → split_hatch is None; use hue_value.
            # - hatch == categorical axis → use the category label.
            if hatch_value is not None:
                hatch_key = hatch_value
            elif hue is not None and hue == hatch:
                hatch_key = hue_value
            else:
                hatch_key = cat
            patch.set_hatch(hatch_map.get(hatch_key, ""))
            patch.set_hatch_linewidth(linewidth)

        patch.set_edgecolor(edgecolor if edgecolor is not None else face_color)

        if idx < len(errorbars):
            errorbars[idx].set_color(
                edgecolor if edgecolor is not None else face_color
            )

def _legend(
        ax: Axes,
        hue: Optional[str],
        hatch: str,
        categorical_axis: str,
        alpha: float,
        linewidth: float,
        color: Optional[str],
        edgecolor: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        hatch_map: Optional[Dict[str, str]],
        kwargs: Optional[Dict] = None,
        legend: Union[bool, Dict] = True,
    ) -> None:
    """Stash LegendEntry objects for the bar plot, then render per-axis
    legends for entries not claimed by a figure-level pp.legend_group.

    Four cases (preserving the original dispatch):
      - hue == hatch            -> 1 combined entry under kind="hue"
      - hue == categorical_axis -> 1 hatch-only entry under kind="hatch"
      - hatch == categorical_axis -> 1 hue-only entry under kind="hue"
      - double split            -> 1 hue entry + 1 hatch entry
    """
    if legend is False:
        return

    flags = resolve_legend_flags(legend)
    kwargs = dict(kwargs or {})
    hue_label = kwargs.pop("hue_label", hue)
    hatch_label = kwargs.pop("hatch_label", hatch)
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="rectangle")

    if hue == hatch:
        # Combined legend for hue and hatch
        if flags["hue"]:
            values = list(palette.keys())
            handles = create_legend_handles(
                labels=values,
                colors=[palette[v] for v in values],
                edgecolors=[edgecolor] * len(values) if edgecolor else None,
                hatches=[hatch_map[v] for v in values],
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=handles,
                    labels=values,
                ),
            )
    elif hue == categorical_axis:
        # Hatch-only legend. Bars are colored by category (via hue=cat) but
        # the hatch swatches represent a different dimension — render them
        # in gray so they read as "this is about the hatch pattern, not the
        # bar color", matching the double-split case where hue and hatch
        # are distinct columns. When ``hatch`` is None, ``hatch_map`` is an
        # empty dict — there's nothing to stash, and rendering an empty
        # entry would produce a blank legend frame.
        if flags["hatch"] and hatch_map:
            labels = list(hatch_map.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=["gray"] * len(hatch_map),
                edgecolors=[edgecolor] * len(hatch_map) if edgecolor else None,
                hatches=list(hatch_map.values()),
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hatch_label,
                    kind="hatch",
                    handles=handles,
                    labels=labels,
                ),
            )
    elif hatch == categorical_axis:
        # Hue-only legend
        if flags["hue"]:
            labels = list(palette.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=[palette[v] for v in palette.keys()],
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                hatches=None,
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=handles,
                    labels=labels,
                ),
            )
    else:
        # Double split: hue first, then hatch
        if flags["hue"] and palette is not None and len(palette) > 0:
            labels = list(palette.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=[palette[v] for v in palette.keys()],
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                hatches=None,
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=handles,
                    labels=labels,
                ),
            )
        if flags["hatch"] and hatch_map is not None and len(hatch_map) > 0:
            # "gray" when hue exists, else resolved color -- matches pre-migration behavior
            hatch_color = "gray" if hue is not None else resolve_param("color", color)
            labels = list(hatch_map.keys())
            handles = create_legend_handles(
                labels=labels,
                colors=[hatch_color] * len(hatch_map),
                edgecolors=[edgecolor] * len(hatch_map) if edgecolor else None,
                hatches=list(hatch_map.values()),
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hatch_label,
                    kind="hatch",
                    handles=handles,
                    labels=labels,
                ),
            )

    render_entries(ax, flags=flags, legend_kws=kwargs)


# =============================================================================
# Stacked / filled bar path (multiple="stack"|"fill")
# =============================================================================


def _validate_gain_levels(
    data: pd.DataFrame,
    stack_col: str,
    categorical_axis: str,
    dodge_col: Optional[str],
) -> None:
    """Validate that ``stack_col`` has exactly 2 unique levels and that
    both levels have at least one observation at every (cat, dodge)
    combination.

    Raises ``ValueError`` on level-count mismatch or on a missing
    (cat, dodge, level) combination — gain semantics are ill-defined
    when a level is absent at some cat, and the caller cannot compute
    a meaningful ``min`` / ``max`` pair.
    """
    levels = _categories_in_draw_order(data[stack_col])
    if len(levels) != 2:
        raise ValueError(
            f"barplot: multiple='gain' requires exactly 2 levels on "
            f"{stack_col!r}; got {len(levels)} ({list(levels)!r})"
        )
    cats = _categories_in_draw_order(data[categorical_axis])
    dodge_levels = (
        _categories_in_draw_order(data[dodge_col])
        if dodge_col is not None else [None]
    )
    for cat in cats:
        for d_lv in dodge_levels:
            for lv in levels:
                mask = data[categorical_axis] == cat
                mask = mask & (data[stack_col] == lv)
                if dodge_col is not None:
                    mask = mask & (data[dodge_col] == d_lv)
                if not mask.any():
                    if dodge_col is not None:
                        raise ValueError(
                            f"barplot: multiple='gain' requires both "
                            f"levels at every (category, dodge) combination; "
                            f"missing level {lv!r} at ({categorical_axis}="
                            f"{cat!r}, {dodge_col}={d_lv!r})"
                        )
                    raise ValueError(
                        f"barplot: multiple='gain' requires both levels "
                        f"at every category; missing level {lv!r} at "
                        f"{categorical_axis}={cat!r}"
                    )


def _draw_stacked(
    *,
    ax: Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    hatch: Optional[str],
    categorical_axis: str,
    split: BarSplitSpec,
    multiple: str,
    stack_by: Optional[str],
    palette: Dict[str, str],
    hatch_map: Dict[str, str],
    color: Optional[str],
    edgecolor: Optional[str],
    linewidth: float,
    errorbar: Optional[str],
    gap: float,
) -> None:
    """Draw stacked (``multiple="stack"``) or 100%-stacked (``multiple="fill"``)
    bars via raw ``ax.bar`` / ``ax.barh`` with cumulative bottom / left per
    stack position.

    Three cases based on which split dimensions are active:

    - **Single-dim stack** (only ``hue`` XOR ``hatch`` distinct from cat):
      one stack per cat position.
    - **Dual-dim with stack_by** (both ``hue`` and ``hatch`` distinct): the
      dimension named in ``stack_by`` stacks; the *other* dimension dodges
      side-by-side within each cat position, producing ``N_other``
      dodged stacks per cat. ``stack_by`` is required here.
    - **``hue == hatch``**: single stack per cat; hatches are overlaid on
      colored swatches (same resolution rules as the dodge path).

    Errorbars are never drawn: per-segment errors aren't additive without
    covariance info and visually collide with the stacked segments above.
    """
    if split.split_hue is None and split.split_hatch is None:
        raise ValueError(
            "barplot: multiple='stack'|'fill' requires either hue= or hatch= "
            "(distinct from the categorical axis) to define the stack "
            "dimension."
        )

    both_active = split.split_hue is not None and split.split_hatch is not None
    if both_active and stack_by not in ("hue", "hatch"):
        raise ValueError(
            "barplot: multiple='stack'|'fill' with distinct hue= and hatch= "
            "columns requires stack_by='hue' or stack_by='hatch' to choose "
            "which dimension stacks (the other will be dodged side-by-side "
            "within each category)."
        )
    if stack_by is not None and stack_by not in ("hue", "hatch"):
        raise ValueError(
            f"barplot: stack_by must be 'hue' or 'hatch'; got {stack_by!r}"
        )

    if errorbar not in (None, False):
        warnings.warn(
            "barplot: errorbars are not drawn with multiple='stack'|'fill' "
            "(per-segment errors are not additive without covariance info); "
            "dropping errorbars for this call.",
            UserWarning,
            stacklevel=3,
        )

    # Decide stack vs dodge dimension. The stack dimension is the one that
    # accumulates along the value axis; the dodge dimension is the one that
    # sub-divides the cat position.
    if both_active:
        stack_is_hue = (stack_by == "hue")
    else:
        stack_is_hue = split.split_hue is not None
    stack_col = split.split_hue if stack_is_hue else split.split_hatch
    dodge_col = split.split_hatch if stack_is_hue else split.split_hue

    if multiple == "gain":
        _validate_gain_levels(
            data=data,
            stack_col=stack_col,
            categorical_axis=categorical_axis,
            dodge_col=dodge_col,
        )

    stack_levels = _categories_in_draw_order(data[stack_col])
    dodge_levels = (
        _categories_in_draw_order(data[dodge_col]) if dodge_col is not None else [None]
    )
    cats = _categories_in_draw_order(data[categorical_axis])
    cat_to_pos = {c: i for i, c in enumerate(cats)}
    value_col = y if categorical_axis == x else x

    # Aggregate mean per (cat, dodge_level, stack_level). A missing
    # combination is omitted — zero-height stubs would still consume a
    # seg in iter order but drawn with height 0 for alignment.
    means: Dict[Tuple[object, Optional[object], object], float] = {}
    for cat in cats:
        for d_lv in dodge_levels:
            for s_lv in stack_levels:
                mask = data[categorical_axis] == cat
                mask = mask & (data[stack_col] == s_lv)
                if dodge_col is not None:
                    mask = mask & (data[dodge_col] == d_lv)
                vals = data.loc[mask, value_col].to_numpy()
                if len(vals):
                    means[(cat, d_lv, s_lv)] = float(vals.mean())

    # Normalize to 100% stacks for multiple="fill" — per dodged stack.
    if multiple == "fill":
        totals: Dict[Tuple[object, Optional[object]], float] = {}
        for (cat, d_lv, _s_lv), m in means.items():
            totals[(cat, d_lv)] = totals.get((cat, d_lv), 0.0) + m
        means = {
            (cat, d_lv, s_lv): (m / totals[(cat, d_lv)]) if totals.get((cat, d_lv)) else 0.0
            for (cat, d_lv, s_lv), m in means.items()
        }

    # Geometry: the cat slot has width 1. Without a dodge dim, one stack
    # of width (1 - gap) centered at the cat position. With a dodge dim of
    # cardinality k, k sub-stacks each of width ((1 - gap) / k) tiled with
    # no inter-sub gap (matches seaborn's dodge layout).
    k = len(dodge_levels)
    slot = max(1.0 - gap, 0.0)
    sub_width = slot / k

    cum: Dict[Tuple[object, Optional[object]], float] = {}

    if multiple == "gain":
        levels = _categories_in_draw_order(data[stack_col])
        l0, l1 = levels[0], levels[1]
        k = len(dodge_levels)
        slot = max(1.0 - gap, 0.0)
        sub_width = slot / k

        for cat in cats:
            for d_idx, d_lv in enumerate(dodge_levels):
                # compute v0, v1 for the 2 stack levels at this (cat, d_lv)
                m0 = data[categorical_axis] == cat
                m0 = m0 & (data[stack_col] == l0)
                if dodge_col is not None:
                    m0 = m0 & (data[dodge_col] == d_lv)
                m1 = data[categorical_axis] == cat
                m1 = m1 & (data[stack_col] == l1)
                if dodge_col is not None:
                    m1 = m1 & (data[dodge_col] == d_lv)
                v0 = float(data.loc[m0, value_col].mean())
                v1 = float(data.loc[m1, value_col].mean())

                pos_cat = cat_to_pos[cat]
                if dodge_col is not None:
                    d_offset = -slot / 2 + sub_width / 2 + d_idx * sub_width
                else:
                    d_offset = 0.0
                pos = pos_cat + d_offset

                if v0 == v1:
                    # Tie: one rect only, colored by levels[0].
                    face = palette.get(l0, color) if palette else color
                    edge = edgecolor if edgecolor is not None else face
                    if split.orient == "v":
                        ax.bar(pos, v0, bottom=0.0, width=sub_width,
                               color=face, edgecolor=edge,
                               linewidth=linewidth)
                    else:
                        ax.barh(pos, v0, left=0.0, height=sub_width,
                                color=face, edgecolor=edge,
                                linewidth=linewidth)
                    continue

                # Distinct: sort into (lo_level, lo_val) + (hi_level, hi_val)
                if v0 < v1:
                    lo_lv, lo_v, hi_lv, hi_v = l0, v0, l1, v1
                else:
                    lo_lv, lo_v, hi_lv, hi_v = l1, v1, l0, v0

                lo_face = palette.get(lo_lv, color) if palette else color
                hi_face = palette.get(hi_lv, color) if palette else color
                lo_edge = edgecolor if edgecolor is not None else lo_face
                hi_edge = edgecolor if edgecolor is not None else hi_face

                if split.orient == "v":
                    ax.bar(pos, lo_v, bottom=0.0, width=sub_width,
                           color=lo_face, edgecolor=lo_edge,
                           linewidth=linewidth)
                    ax.bar(pos, hi_v - lo_v, bottom=lo_v, width=sub_width,
                           color=hi_face, edgecolor=hi_edge,
                           linewidth=linewidth)
                else:
                    ax.barh(pos, lo_v, left=0.0, height=sub_width,
                            color=lo_face, edgecolor=lo_edge,
                            linewidth=linewidth)
                    ax.barh(pos, hi_v - lo_v, left=lo_v, height=sub_width,
                            color=hi_face, edgecolor=hi_edge,
                            linewidth=linewidth)

        # tick + invert logic identical to the stack/fill tail below
        if split.orient == "v":
            ax.set_xticks(list(range(len(cats))))
            ax.set_xticklabels([str(c) for c in cats])
        else:
            ax.set_yticks(list(range(len(cats))))
            ax.set_yticklabels([str(c) for c in cats])
            if ax.get_ylim()[0] < ax.get_ylim()[1]:
                ax.invert_yaxis()
        return

    # Iteration order for double-dim matches the dodge path's draw order
    # (split.iter_draw_order) so the annotate builder pairs patches to
    # aggregates identically. That order is:
    #   hue-outer, hatch-middle, cat-inner
    # ...which for a dodged-stack figure means we draw all segments of
    # a given cat-position at once when possible (this still works for
    # annotate pairing because we compare patch-by-patch against agg
    # rows in the same order).
    for cat, h_val, ht_val in split.iter_draw_order(data):
        if stack_is_hue:
            s_lv, d_lv = h_val, ht_val
        else:
            s_lv, d_lv = ht_val, h_val
        key = (cat, d_lv, s_lv)
        value = means.get(key)
        if value is None:
            continue

        cum_key = (cat, d_lv)
        base = cum.get(cum_key, 0.0)
        pos_cat = cat_to_pos[cat]

        # Sub-position within the cat slot: same ordering as dodge (the
        # dodge dim's first level is leftmost).
        if dodge_col is not None:
            d_idx = dodge_levels.index(d_lv)
            d_offset = -slot / 2 + sub_width / 2 + d_idx * sub_width
        else:
            d_offset = 0.0
        pos = pos_cat + d_offset

        face_color = palette.get(h_val, color) if palette and h_val is not None else color

        # Hatch resolution (mirrors _paint_bars in the dodge path):
        # - split_hatch active → hatch_value authoritative
        # - hue == hatch (single-dim case, split_hatch None) → hue value
        # - otherwise no hatch (or hatch==cat)
        hatch_key: Optional[object] = None
        if split.split_hatch is not None:
            hatch_key = ht_val
        elif hue is not None and hue == hatch:
            hatch_key = h_val
        hatch_pat = hatch_map.get(hatch_key, "") if hatch_map and hatch_key is not None else ""

        edge = edgecolor if edgecolor is not None else face_color

        if split.orient == "v":
            artists = ax.bar(
                pos, value, bottom=base, width=sub_width,
                color=face_color, edgecolor=edge,
                linewidth=linewidth, hatch=hatch_pat,
            )
        else:
            artists = ax.barh(
                pos, value, left=base, height=sub_width,
                color=face_color, edgecolor=edge,
                linewidth=linewidth, hatch=hatch_pat,
            )
        if hatch_pat:
            for patch in artists:
                patch.set_hatch_linewidth(linewidth)

        cum[cum_key] = base + value

    if split.orient == "v":
        ax.set_xticks(list(range(len(cats))))
        ax.set_xticklabels([str(c) for c in cats])
    else:
        ax.set_yticks(list(range(len(cats))))
        ax.set_yticklabels([str(c) for c in cats])
        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()


