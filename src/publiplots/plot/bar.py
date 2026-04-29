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
from publiplots.utils import is_categorical, as_categorical, create_legend_handles, legend
from publiplots.utils.transparency import ArtistTracker

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
    figsize: Optional[Tuple[float, float]] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    hatch_map: Optional[Dict[str, str]] = None,
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    errorbar: str = "se",
    gap: float = 0.1,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    hatch_order: Optional[List[str]] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
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
        Column name for splitting bars side-by-side with hatch patterns.
        When specified, creates grouped bars within each x category.
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
    figsize : tuple, default=(4, 4)
        Figure size (width, height) if creating new figure.
    palette : str, dict, or list, optional
        Color palette. Can be:
        - str: seaborn palette name or publiplots palette name
        - dict: mapping from hue values to colors
        - list: list of colors
    hatch_map : dict, optional
        Mapping from hatch values to hatch patterns.
        Example: {"group1": "", "group2": "///", "group3": "\\\\\\"}
    legend : bool, default=True
        Whether to show the legend.
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
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Simple bar plot:
    >>> fig, ax = pp.barplot(data=df, x="category", y="value")

    Bar plot with color groups:
    >>> fig, ax = pp.barplot(data=df, x="category", y="value",
    ...                       hue="group", palette="pastel")

    Bar plot with hatched bars and patterns:
    >>> fig, ax = pp.barplot(
    ...     data=df, x="condition", y="measurement",
    ...     hatch="treatment", hue="condition",
    ...     hatch_map={"control": "", "treated": "///"},
    ...     palette={"A": "#75b375", "B": "#8e8ec1"}
    ... )

    See Also
    --------
    barplot_enrichment : Specialized bar plot for enrichment analysis
    """
    # Read defaults from rcParams if not provided
    figsize = resolve_param("figure.figsize", figsize)
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    capsize = resolve_param("capsize", capsize)
    color = resolve_param("color", color)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
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

    # Determine the strategy for handling hatch and hue
    # Key insight: use hue=hatch for splitting when needed, then override colors
    sns_hue = hue
    sns_palette = palette

    # When hue == categorical axis, seaborn doesn't need hue for splitting —
    # it would only cause unwanted dodge/grouping. Color by palette via x instead.
    if hue is not None and hue == categorical_axis:
        sns_hue = None
        sns_palette = None

    # hatch split only needed if hatch is not the same as the categorical axis
    split_by_hatch = hatch is not None and hatch != categorical_axis
    double_split = split_by_hatch and hue is not None and hue != categorical_axis and hatch != hue
    if split_by_hatch:
        if double_split:
            # Need to create double split by creating a new column with the combined value of hue and hatch
            sns_hue = f"{hue}_{hatch}"
            # Color bars by 
            sns_palette = {
                x: palette[x.split(_SPLIT_SEPARATOR)[0]] for x in data[f"{hue}_{hatch}"].cat.categories
            }
        else:
            # Only need to split by hatch
            # We will recolor the bars to color argument if hue is None
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

    # When hue == categorical axis, recolor bars from palette
    if hue is not None and hue == categorical_axis and palette:
        categories = order if order else list(data[categorical_axis].cat.categories)
        for idx, patch in enumerate(tracker.get_new_patches()):
            if idx < len(categories):
                bar_color = palette.get(categories[idx], color)
                # Set edge to palette color so the main flow copies it to face correctly
                patch.set_edgecolor(bar_color)
        # Also recolor error bar lines
        for idx, line in enumerate(tracker.get_new_lines()):
            if idx < len(categories):
                line.set_color(edgecolor if edgecolor is not None else palette.get(categories[idx], color))

    # Apply hatch patterns and override colors if needed
    if hatch is not None:
        _apply_hatches_and_override_colors(
            ax=ax,
            data=data,
            hue=hue,
            hatch=hatch,
            categorical_axis=categorical_axis,
            double_split=double_split,
            linewidth=linewidth,
            color=color,
            edgecolor=edgecolor,
            palette=palette,
            hatch_map=hatch_map,
        )

    # Set facecolor from seaborn's edge (palette color), optionally override edgecolor
    for patch in tracker.get_new_patches():
        palette_color = patch.get_edgecolor()  # seaborn fill=False puts palette color on edge
        patch.set_facecolor(palette_color)
        if edgecolor is not None:
            patch.set_edgecolor(edgecolor)
    # Apply differential transparency to face vs edge
    tracker.apply_transparency(on="patches", face_alpha=alpha, edge_alpha=1.0)

    # Apply edgecolor to error bar lines
    if edgecolor is not None:
        for line in tracker.get_new_lines():
            line.set_color(edgecolor)

    # Add legend if hue or hatch is used
    if legend:
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
        )

    # Set labels
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    return fig, ax


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

def _apply_hatches_and_override_colors(
        ax: Axes,
        data: pd.DataFrame,
        hue: Optional[str],
        hatch: str,
        categorical_axis: str,
        double_split: bool,
        linewidth: float,
        color: Optional[str],
        edgecolor: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        hatch_map: Optional[Dict[str, str]],
    ) -> None:
    """
    Apply hatch patterns using patch.get_label() and idx // n_x, then override colors.

    Uses the approach from user"s example:
    - Track which category each patch belongs to via idx // n_x
    - Use patch.get_label() or hue_order to determine the category
    - Apply hatch based on the hatch column value
    - Override colors if needed
    """
    # Override color if hue is not defined (default to color argument)
    # Or if hue is not the same as hatch
    override_color = hue is None or hue != hatch
    n_axis = len(data[categorical_axis].unique())
    n_hue = len(data[hue].unique()) if hue is not None else 1
    n_hatch = len(data[hatch].unique())  # always defined
    total_bars = n_axis * n_hue * n_hatch
    hue_order = list(palette.keys())
    hatch_order = list(hatch_map.keys())
    errorbars = ax.get_lines()

    for idx, patch in enumerate(ax.patches):
        if not hasattr(patch, "get_height"):
            continue

        bar_idx = idx % total_bars
        axis_idx = bar_idx % n_axis
        # Get hatch and hue index in palette and hatch_map
        # If hatch is the same as the categorical axis, we need to use the axis_idx
        hatch_idx = (bar_idx // n_axis) % n_hatch if hatch != categorical_axis else axis_idx
        # If double split, we take into account the combined index
        # Otherwise, if matching hatch and hue, we use the hatch index
        # Otherwise, we use the axis index
        hue_idx = (bar_idx // (n_axis * n_hatch)) if double_split else (
            hatch_idx if hue == hatch else axis_idx
        )

        # Apply hatch pattern to all patches
        hatch_pattern = hatch_map.get(hatch_order[hatch_idx], "")
        patch.set_hatch(hatch_pattern)
        # set_hatch_linewidth
        patch.set_hatch_linewidth(linewidth)

        # Repaint the bars when needed (override colors if not using double split)
        bar_color = palette[hue_order[hue_idx]] if hue is not None else color
        if not (double_split or hatch == categorical_axis):
            # Use the same color for all bars
            patch.set_edgecolor(edgecolor if edgecolor is not None else bar_color)
            patch.set_facecolor(bar_color)

            # Match error bar colors to bar colors
            if bar_idx < len(errorbars):
                errorbars[bar_idx].set_color(edgecolor if edgecolor is not None else bar_color)

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
    ) -> None:
    """
    Create legend handles for bar plot.
    If hue or hatch is the same as the categorical axis, we can skip corresponding legend.
    If matching hatch and hue, we need to create a legend for the combined hue and hatch.
    If double split, we need to create a legend for the hue and hatch separately.
    Otherwise, we need to create a legend for the hue and hatch (same as double split)
    """
    kwargs = kwargs or {}

    builder = legend(ax=ax, auto=False)

    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="rectangle")
    hue_label = kwargs.pop("hue_label", hue)
    hatch_label = kwargs.pop("hatch_label", hatch)

    if hue == hatch:
        # combined legend for hue and hatch
        values = list(palette.keys())
        builder.add_legend(
            handles=create_legend_handles(
                labels=values,
                colors=[palette[v] for v in values],
                edgecolors=[edgecolor] * len(values) if edgecolor else None,
                hatches=[hatch_map[v] for v in values],
                **handle_kwargs
            ),
            label=hue_label,
        )

    elif hue == categorical_axis:
        # legend for hatch only
        builder.add_legend(
            handles=create_legend_handles(
                labels=list(hatch_map.keys()),
                colors=[resolve_param("color", color)] * len(hatch_map),
                edgecolors=[edgecolor] * len(hatch_map) if edgecolor else None,
                hatches=list(hatch_map.values()),
                **handle_kwargs
            ),
            label=hatch_label,
        )

    elif hatch == categorical_axis:
        # legend for hue only
        builder.add_legend(
            handles=create_legend_handles(
                labels=list(palette.keys()),
                colors=[palette[v] for v in palette.keys()],
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                hatches=None,
                **handle_kwargs
            ),
            label=hue_label,
        )
    else:
        # legend for hue and hatch separately (DOUBLE SPLIT)
        # Add hue legend first
        if palette is not None and len(palette) > 0:
            builder.add_legend(
                handles=create_legend_handles(
                    labels=list(palette.keys()),
                    colors=[palette[v] for v in palette.keys()],
                    edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                    hatches=None,
                    **handle_kwargs
                ),
                label=hue_label,
            )

        # Add hatch legend second
        if hatch_map is not None and len(hatch_map) > 0:
            # Use gray for hatch legend if hue exists, otherwise use color
            hatch_color = "gray" if hue is not None else resolve_param("color", color)
            builder.add_legend(
                handles=create_legend_handles(
                    labels=list(hatch_map.keys()),
                    colors=[hatch_color] * len(hatch_map),
                    edgecolors=[edgecolor] * len(hatch_map) if edgecolor else None,
                    hatches=list(hatch_map.values()),
                    **handle_kwargs
                ),
                label=hatch_label,
            )