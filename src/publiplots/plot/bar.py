"""
Bar plot functions for publiplots.

This module provides publication-ready bar plot visualizations with
flexible styling and grouping options.
"""

from typing import Optional, List, Dict, Tuple, Union

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
from publiplots.annotate._builders import build_from_barplot_call

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
    **kwargs
) -> Axes:
    """
    Create a publication-ready bar plot.

    This function creates bar plots with optional grouping, error bars,
    and hatch patterns. Supports both simple and complex bar plots with
    side-by-side grouped bars.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x : str
        Column name for x-axis categories.
    y : str
        Column name for y-axis values.
    hue : str, optional
        Column name for color grouping (typically same as x for hatched bars).
    hatch : str, optional
        Column name that drives a **second categorical dimension**, rendered
        via hatch textures instead of color. Think of it as a texture-based
        analogue of ``hue``: bars split side-by-side within each x category,
        each assigned a distinct hatch pattern. Frequently combined with
        ``hue`` to encode two categoricals at once (e.g. ``hue='condition'``
        + ``hatch='treatment'``).

        Patterns are drawn from :data:`publiplots.HATCH_PATTERNS` by default;
        density is controlled globally via :func:`publiplots.set_hatch_mode`
        (``"dense"``, ``"sparse"``, or ``"off"``) or overridden per-plot with
        ``hatch_map``. Call :func:`publiplots.list_hatch_patterns` to see the
        catalog.
    color : str, optional
        Fixed color for all bars (only used when hue is None).
        Overrides default color. Example: "#ff0000" or "red".
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label. If empty and hatch is used, uses x column name.
    ylabel : str, default=""
        Y-axis label. If empty, uses y column name.
    linewidth : float, default=1.0
        Width of bar edges.
    capsize : float, default=0.0
        Width of error bar caps.
    alpha : float, default=0.1
        Transparency of bar fill (0-1). Use 0 for outlined bars only.
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
        - str: seaborn palette name or publiplots palette name
        - dict: mapping from hue values to colors
        - list: list of colors
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
        Whether to show the legend. Accepts ``bool`` or
        ``dict[kind, bool]`` for per-kind control (e.g.,
        ``legend={"hatch": False}`` hides just the hatch leg in a
        double-split bar plot).
    annotate : bool or dict, optional
        If True, label each bar with its aggregated value. Pass a dict to
        forward options to pp.annotate (e.g. {"fmt": ".3f", "anchor": "inside"}).
        See :func:`publiplots.annotate` for all supported options.
    errorbar : str, default="se"
        Error bar type: "se" (standard error), "sd" (standard deviation),
        "ci" (confidence interval), or None for no error bars.
    gap : float, default=0.1
        Gap between bar groups (0-1).
    order : list, optional
        Order of x/y-axis categories. If provided, determines bar order.
    hue_order : list, optional
        Hue order. If provided, determines bar order within groups.
    hatch_order : list, optional
        Order of hatch categories. If provided, determines bar order within groups.
    **kwargs
        Additional keyword arguments passed to seaborn.barplot().

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple bar plot:
    >>> ax = pp.barplot(data=df, x="category", y="value")

    Bar plot with color groups:
    >>> ax = pp.barplot(data=df, x="category", y="value",
    ...                  hue="group", palette="pastel")

    Bar plot with hatched bars and patterns:
    >>> ax = pp.barplot(
    ...     data=df, x="condition", y="measurement",
    ...     hatch="treatment", hue="condition",
    ...     hatch_map={"control": "", "treated": "///"},
    ...     palette={"A": "#75b375", "B": "#8e8ec1"}
    ... )

    See Also
    --------
    publiplots.set_hatch_mode : Set the global hatch-density mode.
    publiplots.list_hatch_patterns : Print the built-in hatch patterns.
    publiplots.annotate : Add value labels to bars (see ``annotate=`` above).
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
    from publiplots.annotate._splits import BarSplitSpec
    _split = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=hatch, categorical_axis=categorical_axis,
    )
    split_by_hatch = _split.split_hatch is not None
    double_split = _split.split_hue is not None and _split.split_hatch is not None

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
