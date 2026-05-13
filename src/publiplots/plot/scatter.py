"""
Scatterplot visualization module for publiplots.

Provides flexible scatterplot visualizations with support for both continuous
and categorical data, size encoding, and color encoding (categorical or continuous).
"""

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, List

from publiplots.themes.rcparams import resolve_param

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils import is_categorical, is_numeric, create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)
from publiplots.utils.plot_legend import (
    get_size_ticks,
    merge_categorical_entries,
    render_entries,
    resolve_size_map,
    stash_continuous_hue,
    resolve_style_maps,
)


def scatterplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: Optional[str] = None,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    sizes: Optional[Tuple[float, float]] = None,
    markers: Optional[Union[bool, List[str], Dict[str, str]]] = None,
    size_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    hue_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    background_marker: Union[bool, str] = False,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    margins: Union[float, Tuple[float, float]] = 0.1,
    **kwargs
) -> Axes:
    """
    Create a scatterplot with publiplots styling.

    This function creates scatterplots for both continuous and categorical data
    with extensive customization options. Supports size and color encoding,
    with distinctive double-layer markers for enhanced visibility.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing x, y, and optional size/hue columns.
    x : str
        Column name for x-axis values (continuous or categorical).
    y : str
        Column name for y-axis values (continuous or categorical).
    size : str, optional
        Column name for marker sizes. If None, all markers have the same size.
    hue : str, optional
        Column name for marker colors. Can be categorical or continuous.
        If None, uses default color or the value from `color` parameter.
    style : str, optional
        Column name for marker styles. Produces points with different markers.
        Only categorical data is supported. If None, all markers use the same style.
    color : str, optional
        Fixed color for all markers (only used when hue is None).
        Overrides default color. Example: "#ff0000" or "red".
    palette : str, dict, list, or None
        Color palette for hue values:
        - str: palette name (e.g., "viridis", "pastel")
        - dict: mapping of hue values to colors (categorical only)
        - list: list of colors
        - None: uses default palette
    sizes : tuple of float, optional
        (min_size, max_size) in points^2 for marker sizes.
        Default: (20, 200) for continuous data, (100, 100) for no size encoding.
    markers : bool, list, dict, optional
        Markers to use for different levels of the style variable:
        - True: use default marker set
        - list: list of marker symbols (e.g., ["o", "^", "s"])
        - dict: mapping of style values to markers (e.g., {"A": "o", "B": "^"})
        - None: uses default marker "o" for all points
    size_norm : tuple of float, optional
        (vmin, vmax) for size normalization. If None, computed from data.
    hue_norm : tuple of float, optional
        (vmin, vmax) for continuous hue normalization. If None, computed from data.
        Providing this enables continuous color mapping.
    alpha : float, optional
        Transparency level for marker fill (0-1). When None, resolved from
        ``publiplots.rcParams["alpha"]``.
    linewidth : float, optional
        Width of marker edges. When None, resolved from
        ``publiplots.rcParams["lines.linewidth"]``.
    edgecolor : str, optional
        Color for marker edges. If None, uses same color as fill. Can also be
        set globally via ``publiplots.rcParams["edgecolor"]``.
    background_marker : bool or str, default=False
        Publiplots-specific: draw a solid background-colored marker behind
        each point to hide overlap. ``True`` uses white; a color string
        (e.g. ``"#eeeeee"``) overrides the background color. Off by default
        because duplicating every point doubles artist count and overlap is
        often informative.
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label. If empty, uses x column name.
    ylabel : str, default=""
        Y-axis label. If empty, uses y column name.
    legend_kws : dict, optional
        Keyword arguments for legend builder:

        - hue_title : str, optional - Title for the hue legend. If None, uses hue column name.
        - size_title : str, optional - Title for the size legend. If None, uses size column name.
        - style_title : str, optional - Title for the style legend. If None, uses style column name.
        - size_reverse : bool, default=True - Whether to reverse the size legend (descending order).
    legend : bool, default=True
        Whether to show legend.
    margins : float or tuple, default=0.1
        Margins around the plot for categorical axes. 
        If a float, sets both x and y margins to the same value.
        If a tuple, sets x and y margins separately.
    **kwargs
        Additional keyword arguments passed to seaborn.scatterplot().

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple scatterplot with continuous data:
    >>> ax = pp.scatterplot(data=df, x="time", y="value")

    Scatterplot with size encoding:
    >>> ax = pp.scatterplot(data=df, x="time", y="value",
    ...                      size="magnitude", sizes=(20, 200))

    Scatterplot with categorical color encoding:
    >>> ax = pp.scatterplot(data=df, x="time", y="value",
    ...                      hue="group", palette="pastel")

    Scatterplot with continuous color encoding:
    >>> ax = pp.scatterplot(data=df, x="time", y="value",
    ...                      hue="score", palette="viridis",
    ...                      hue_norm=(0, 100))

    Scatterplot with custom single color:
    >>> ax = pp.scatterplot(data=df, x="time", y="value",
    ...                      color="#e67e7e")

    Scatterplot with different marker styles:
    >>> ax = pp.scatterplot(data=df, x="time", y="value",
    ...                      hue="group", style="condition",
    ...                      markers=["o", "^", "s"])

    Categorical scatterplot (positions on grid):
    >>> ax = pp.scatterplot(data=df, x="category", y="condition",
    ...                      size="pvalue", hue="log2fc")

    Dense overlapping data with background markers for visual contrast:
    >>> ax = pp.scatterplot(data=df, x="umap1", y="umap2",
    ...                      hue="cluster", background_marker=True)

    See Also
    --------
    publiplots.heatmap : Dot/bubble heatmap mode uses scatterplot internally.
    publiplots.pointplot : Aggregated point estimates with error bars.
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Validate required columns
    required_cols = [x, y]
    if size is not None:
        required_cols.append(size)
    if hue is not None:
        required_cols.append(hue)
    if style is not None:
        required_cols.append(style)

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # Copy data to avoid modifying original
    data = data.copy()

    # Create figure via pp.subplots to install SubplotsAutoLayout; users who
    # want custom dimensions should compose with pp.subplots(axes_size=...)
    # before calling and pass ax=.
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Determine if x and y are categorical
    x_is_categorical = is_categorical(data[x])
    y_is_categorical = is_categorical(data[y])

    # Handle categorical positioning
    if x_is_categorical or y_is_categorical:
        data, x_col, y_col, x_labels, y_labels = _handle_categorical_axes(
            data, x, y, x_is_categorical, y_is_categorical
        )
    else:
        x_col, y_col = x, y
        x_labels, y_labels = None, None

    # Determine color/palette to use
    color = resolve_param("color", color)
    palette = resolve_palette_map(
        values=data[hue].unique() if hue is not None else None,
        palette=palette,
    ) if hue is not None else None

    # Set default size range for size-mapped plots. The unsized case is
    # handled below via the top-level ``s=`` kwarg (seaborn ignores
    # ``sizes=`` when ``size`` is None — that's why a ``sizes=(100, 100)``
    # default here used to be silently dropped, with seaborn's own 36-pt²
    # fallback taking effect instead).
    if sizes is None:
        sizes = (20, 200)
    # Unified marker area, in points², for the unsized case. lines.markersize
    # is a diameter in points; matplotlib scatter takes points². Square it
    # so a single rcParams knob drives every marker-bearing plot.
    default_markersize = resolve_param("lines.markersize", None)
    default_marker_area = float(default_markersize) ** 2

    # Size resolution: numeric → Normalize + get_size_ticks; categorical →
    # explicit {category: area_points²} map (get_size_ticks assumes numeric).
    size_is_numeric = size is not None and is_numeric(data[size])
    size_map: Dict = {}
    if size is not None and size_is_numeric:
        if size_norm is None:
            size_norm = (data[size].min(), data[size].max())
        if isinstance(size_norm, tuple):
            size_norm = Normalize(vmin=size_norm[0], vmax=size_norm[1])
    elif size is not None:
        size_map = resolve_size_map(
            size=size, data=data, size_order=None, sizes=sizes,
        )

    # Create normalization for hue if needed
    if hue is not None and is_numeric(data[hue]):
        if hue_norm is None:
            hue_norm = (data[hue].min(), data[hue].max())
        if isinstance(hue_norm, tuple):
            hue_norm = Normalize(vmin=hue_norm[0], vmax=hue_norm[1])

    # If style is provided without markers, use default markers
    if style is not None and markers is None:
        markers = True

    # Prepare kwargs for seaborn scatterplot
    scatter_kwargs = {
        "data": data,
        "x": x_col,
        "y": y_col,
        "hue": hue,
        "hue_norm": hue_norm,
        "size": size,
        # Categorical → forward the resolved {category: area} map so seaborn
        # and the legend agree on per-category marker areas.
        "sizes": (size_map if size_map else sizes) if size is not None else None,
        "size_norm": size_norm if size is not None else None,
        "style": style,
        "markers": markers if style is not None else None,
        "ax": ax,
        "color": color,
        "palette": palette,
        "legend": False,
    }
    # Unsized case: seaborn ignores ``sizes=`` and falls back to its own
    # default (36 pt²). Forward a top-level ``s=`` so publiplots' rcParam
    # actually drives the marker area.
    if size is None:
        scatter_kwargs["s"] = default_marker_area

    # Merge with user kwargs
    scatter_kwargs.update(kwargs)

    # Create scatter plot with edges
    scatter_kwargs.update({
        "linewidth": linewidth,
        "zorder": 2,
    })
    from publiplots.utils.transparency import (
        ArtistTracker,
        apply_transparency,
        apply_background_markers,
        composite_facecolors_over,
    )
    tracker = ArtistTracker(ax)
    sns.scatterplot(**scatter_kwargs)

    new_collections = tracker.get_new_collections()
    collection = new_collections[0]

    # When background_marker is active, we can't rely on zorder alone: all
    # points within a PathCollection share the same zorder and alpha-blend
    # together in a single paint pass, ignoring any disc below. Instead we
    # pre-composite the face color over bg_color and draw the foreground at
    # full opacity — so overlapping points last-draw-wins instead of blending.
    bg_color = None
    if background_marker:
        bg_color = "white" if background_marker is True else background_marker

    for c in new_collections:
        c.set_edgecolors(
            edgecolor if edgecolor else c.get_facecolors()
        )
        c.set_linewidths(linewidth)
        if bg_color is not None:
            composite_facecolors_over(c, bg_color, alpha=alpha)
            apply_transparency(c, face_alpha=1.0, edge_alpha=1.0)
        else:
            # Apply differential transparency to face vs edge
            apply_transparency(c, face_alpha=alpha, edge_alpha=1.0)

    # Optional: solid background twins below the foreground. Not strictly
    # needed for overlap hiding (handled by the composite above) but it keeps
    # the edge ring and marker shape on a solid ground if the axes have a
    # non-white patch color.
    if bg_color is not None:
        apply_background_markers(
            new_collections, ax, background_color=bg_color,
        )

    # Set colormap and normalization for collection
    # Used by legend builder to create colorbar
    if hue is not None:
        collection.set_label(hue)
        if hue_norm is not None:
            collection.set_cmap(palette)  # is a string or cmap
            collection.set_norm(hue_norm)

    # Handle categorical axis labels
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    # Set labels and title
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    # Stash legend entries (per-kind) and optionally render per-axis legends.
    # legend may be bool or dict[str, bool]; False short-circuits both stash
    # and render, True stashes/renders everything, and a dict filters per kind.
    _legend(
        ax=ax,
        data=data,
        hue=hue,
        size=size,
        style=style,
        markers=markers,
        size_map=size_map,
        color=color,
        palette=palette,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        hue_norm=hue_norm,
        size_norm=size_norm,
        sizes=sizes,
        kwargs=legend_kws,
        legend=legend,
    )

    # Set margins for categorical axes automatically
    if x_is_categorical or y_is_categorical:
        if isinstance(margins, (float, int)):
            margins = (margins, margins)
        ax.margins(
            x=margins[0] if x_is_categorical else None, 
            y=margins[1] if y_is_categorical else None
        )

    return ax

def _handle_categorical_axes(
    data: pd.DataFrame,
    x: str,
    y: str,
    x_is_categorical: bool,
    y_is_categorical: bool
) -> Tuple[pd.DataFrame, str, str, Optional[List], Optional[List]]:
    """
    Handle categorical axes by creating position mappings.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        X column name.
    y : str
        Y column name.
    x_is_categorical : bool
        Whether x is categorical.
    y_is_categorical : bool
        Whether y is categorical.

    Returns
    -------
    data : pd.DataFrame
        Data with added position columns.
    x_col : str
        Column name to use for x plotting.
    y_col : str
        Column name to use for y plotting.
    x_labels : list or None
        X-axis labels if categorical.
    y_labels : list or None
        Y-axis labels if categorical.
    """
    data = data.copy()

    if x_is_categorical:
        x_cats = data[x].unique()
        x_positions = {cat: i for i, cat in enumerate(x_cats)}
        data["_x_pos"] = data[x].map(x_positions)
        x_col = "_x_pos"
        x_labels = x_cats
    else:
        x_col = x
        x_labels = None

    if y_is_categorical:
        y_cats = data[y].unique()
        y_positions = {cat: i for i, cat in enumerate(y_cats)}
        data["_y_pos"] = data[y].map(y_positions)
        y_col = "_y_pos"
        y_labels = y_cats
    else:
        y_col = y
        y_labels = None

    return data, x_col, y_col, x_labels, y_labels

def _legend(
        ax: Axes,
        data: pd.DataFrame, # for size legend
        hue: Optional[str],
        size: Optional[str],
        style: Optional[str],
        markers: Optional[Union[bool, List[str], Dict[str, str]]],
        color: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        hue_norm: Optional[Normalize],
        size_norm: Optional[Normalize],
        sizes: Tuple[float, float],
        size_map: Optional[Dict] = None,
        alpha: Optional[float] = None,
        linewidth: Optional[float] = None,
        edgecolor: Optional[str] = None,
        kwargs: Optional[Dict] = None,
        legend: Union[bool, Dict] = True,
    ) -> None:
    """
    Stash LegendEntry objects for scatter plot and optionally render per-axis legends.

    Parameters
    ----------
    legend : bool or dict, default=True
        - ``True``  -> stash all kinds and render per-axis (unless claimed by a group).
        - ``False`` -> stash nothing and render nothing.
        - ``dict``  -> per-kind flags (missing keys default to True).
    """
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)

    kwargs = kwargs or {}
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="circle")

    flags = resolve_legend_flags(legend)

    # Detect "same variable used for multiple kinds" so we can stash one
    # merged LegendEntry instead of N duplicate columns. Merge requires
    # both dimensions to be categorical (continuous hue is a colorbar,
    # which can't composite with shaped markers).
    categorical_hue = hue is not None and isinstance(palette, dict)
    # Resolve style marker map up-front (used by the merge path and the
    # independent style-stash path below).
    style_values = list(data[style].unique()) if style is not None else []
    if style is not None:
        marker_map, _ = resolve_style_maps(
            style=style,
            data=data,
            style_order=style_values,
            markers=markers,
            dashes=None,
        )
    else:
        marker_map = {}
    categorical_style = style is not None and bool(marker_map)
    merge_hue_style = (
        categorical_hue
        and categorical_style
        and hue == style
        and flags["hue"]
        and flags["style"]
    )

    # Stash hue entry
    if hue is not None and flags["hue"]:
        hue_label = kwargs.pop("hue_label", hue)
        if isinstance(palette, dict):  # categorical
            hue_labels = list(palette.keys())
            if merge_hue_style:
                # Combined swatch: colored marker of the right shape, per row.
                merged = merge_categorical_entries(
                    name=hue_label,
                    labels=[str(v) for v in hue_labels],
                    colors=list(palette.values()),
                    markers=[marker_map[v] for v in hue_labels],
                    edgecolors=(
                        [edgecolor] * len(hue_labels) if edgecolor else None
                    ),
                    handle_kwargs=handle_kwargs,
                )
                stash_entry(ax, merged)
                kwargs.pop("style_label", None)
            else:
                hue_handles = create_legend_handles(
                    labels=hue_labels,
                    colors=list(palette.values()),
                    edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                    **handle_kwargs
                )
                stash_entry(
                    ax,
                    LegendEntry.build(
                        name=hue_label,
                        kind="hue",
                        handles=hue_handles,
                        labels=hue_labels,
                    ),
                )
        else:
            # continuous -> colorbar
            stash_continuous_hue(
                ax, name=hue_label, palette=palette, hue_norm=hue_norm
            )

    # Stash size entry. Numeric → MaxNLocator ticks + area-to-markersize
    # via get_size_ticks. Categorical → one marker per category, marker
    # diameter derived from the category's area (same sqrt(a/pi)*2 rule).
    if size is not None and flags["size"]:
        tick_color = color if hue is None else "gray"
        size_handle_kwargs = handle_kwargs.copy()
        size_handle_kwargs["color"] = tick_color
        size_label = kwargs.pop("size_label", size)
        if size_map:
            import numpy as _np
            labels = [str(k) for k in size_map.keys()]
            tick_sizes = [_np.sqrt(float(a) / _np.pi) * 2 for a in size_map.values()]
            size_handles = create_legend_handles(
                labels=labels,
                sizes=tick_sizes,
                **size_handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=size_label,
                    kind="size",
                    handles=size_handles,
                    labels=labels,
                ),
            )
        else:
            tick_labels, tick_sizes = get_size_ticks(
                values=data[size].dropna().values,
                sizes=sizes,
                size_norm=size_norm,
                nbins=kwargs.pop("size_nbins", 4),
                min_n_ticks=kwargs.pop("size_min_n_ticks", 3),
                include_min_max=kwargs.pop("size_include_min_max", False),
            )
            size_handles = create_legend_handles(
                labels=tick_labels,
                sizes=tick_sizes,
                **size_handle_kwargs
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=size_label,
                    kind="size",
                    handles=size_handles,
                    labels=tick_labels,
                ),
            )

    # Stash style entry (skipped when already merged into the hue entry above)
    if style is not None and flags["style"] and not merge_hue_style:
        style_label = kwargs.pop("style_label", style)

        # Determine color for style legend
        style_color = color if hue is None else "gray"
        style_handle_kwargs = handle_kwargs.copy()
        style_handle_kwargs["color"] = style_color
        style_handle_kwargs.pop("style", None)  # Remove style key from handle_kwargs

        # Create legend handles with different markers
        style_labels = [str(val) for val in style_values]
        style_handles = create_legend_handles(
            labels=style_labels,
            markers=[marker_map[val] for val in style_values],
            edgecolors=[edgecolor] * len(style_values) if edgecolor else None,
            **style_handle_kwargs
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=style_label,
                kind="style",
                handles=style_handles,
                labels=style_labels,
            ),
        )

    # Render per-axis legends for entries not claimed by a figure-level group.
    # legend=False short-circuits (all flags False, nothing stashed, nothing to render).
    if legend is False:
        return

    render_entries(ax, flags=flags, legend_kws=kwargs)