"""
Scatterplot visualization module for publiplots.

Provides flexible scatterplot visualizations with support for both continuous
and categorical data, size encoding, and color encoding (categorical or continuous).
"""

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, List

from publiplots.themes.rcparams import resolve_param

from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.markers import resolve_marker_map
from publiplots.utils import is_categorical, is_numeric, create_legend_handles, legend


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
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    margins: Union[float, Tuple[float, float]] = 0.1,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
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
        Default: (50, 500) for continuous data, (100, 100) for no size encoding.
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
    alpha : float, default=0.1
        Transparency level for marker fill (0-1).
    linewidth : float, default=2.0
        Width of marker edges.
    edgecolor : str, optional
        Color for marker edges. If None, uses same color as fill.
    figsize : tuple, default=(6, 4)
        Figure size (width, height) if creating new figure.
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
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Simple scatterplot with continuous data:
    >>> fig, ax = pp.scatterplot(data=df, x="time", y="value")

    Scatterplot with size encoding:
    >>> fig, ax = pp.scatterplot(data=df, x="time", y="value",
    ...                           size="magnitude", sizes=(50, 500))

    Scatterplot with categorical color encoding:
    >>> fig, ax = pp.scatterplot(data=df, x="time", y="value",
    ...                           hue="group", palette="pastel")

    Scatterplot with continuous color encoding:
    >>> fig, ax = pp.scatterplot(data=df, x="time", y="value",
    ...                           hue="score", palette="viridis",
    ...                           hue_norm=(0, 100))

    Scatterplot with custom single color:
    >>> fig, ax = pp.scatterplot(data=df, x="time", y="value",
    ...                           color="#e67e7e")

    Scatterplot with different marker styles:
    >>> fig, ax = pp.scatterplot(data=df, x="time", y="value",
    ...                           hue="group", style="condition",
    ...                           markers=["o", "^", "s"])

    Categorical scatterplot (positions on grid):
    >>> fig, ax = pp.scatterplot(data=df, x="category", y="condition",
    ...                           size="pvalue", hue="log2fc")
    """
    # Read defaults from rcParams if not provided
    figsize = resolve_param("figure.figsize", figsize)
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)

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

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
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

    # Set default sizes
    if sizes is None:
        sizes = (100, 100) if size is None else (50, 500)

    # Set default size normalization
    if size is not None and is_numeric(data[size]):
        if size_norm is None:
            size_norm = (data[size].min(), data[size].max())
        if isinstance(size_norm, tuple):
            size_norm = Normalize(vmin=size_norm[0], vmax=size_norm[1])

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
        "sizes": sizes if size is not None else None,
        "size_norm": size_norm if size is not None else None,
        "style": style,
        "markers": markers if style is not None else None,
        "ax": ax,
        "color": color,
        "palette": palette,
        "legend": False,
    }

    # Merge with user kwargs
    scatter_kwargs.update(kwargs)

    # Create scatter plot with edges
    scatter_kwargs.update({
        "linewidth": linewidth,
        "zorder": 2,
    })
    sns.scatterplot(**scatter_kwargs)

    collection = ax.collections[0]
    
    collection.set_edgecolors(
        edgecolor if edgecolor else collection.get_facecolors()
    )
    collection.set_linewidths(linewidth)

    # Apply differential transparency to face vs edge
    from publiplots.utils.transparency import apply_transparency
    apply_transparency(collection, face_alpha=alpha, edge_alpha=1.0)

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

    # Always store legend metadata, but only create legend if legend=True
    _legend(
        ax=ax,
        data=data,
        hue=hue,
        size=size,
        style=style,
        markers=markers,
        color=color,
        palette=palette,
        alpha=alpha,
        linewidth=linewidth,
        hue_norm=hue_norm,
        size_norm=size_norm,
        sizes=sizes,
        kwargs=legend_kws,
        create_legend=legend,  # Only create if legend=True
    )

    # Set margins for categorical axes automatically
    if x_is_categorical or y_is_categorical:
        if isinstance(margins, (float, int)):
            margins = (margins, margins)
        ax.margins(
            x=margins[0] if x_is_categorical else None, 
            y=margins[1] if y_is_categorical else None
        )

    return fig, ax

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

def _get_size_ticks(
        values: np.ndarray,
        sizes: Tuple[float, float],
        size_norm: Normalize,
        nbins: int = 4,
        min_n_ticks: int = 3,
        include_min_max: bool = False,
    ) -> Tuple[List[str], List[float]]:
    """
    Get size ticks for size legend.
    Uses MaxNLocator to generate ticks.
    Includes actual min and max from data.
    Rounds to reasonable precision.
    Falls back to min and max if no ticks are generated.

    Parameters
    ----------
    values : np.ndarray
        Values to get size ticks for.
    sizes : Tuple[float, float]
        (min_size, max_size) in points^2.
    size_norm : Normalize
        Size normalization object.
    nbins : int, default=4
        Number of bins used by MaxNLocator.
    min_n_ticks : int, default=3
        Minimum number of ticks to generate.
    include_min_max : bool, default=False
        Whether to include actual min and max from data.
        If True, the first and last tick will be the actual min and max.
        This is useful if the data is very small and the min and max are not representive.
    Returns
    -------
    tick_labels : List[str]
        Tick labels.
    tick_sizes : List[float]
        Tick sizes.
    """
    unique_vals = np.unique(values[~np.isnan(values)])
    v_min, v_max = size_norm.vmin, size_norm.vmax
    
    if len(unique_vals) <= 4:
        # If few unique values, show them all
        ticks = unique_vals
    else:
        # Use MaxNLocator but ensure we capture extremes
        locator = MaxNLocator(nbins=nbins, min_n_ticks=min_n_ticks)
        ticks = locator.tick_values(v_min, v_max)
        ticks = ticks[(ticks >= v_min) & (ticks <= v_max)]
        
        # Include actual min and max from data
        if include_min_max:
            ticks = np.unique(np.concatenate([[v_min], ticks, [v_max]]))
    
    # Round to reasonable precision
    if v_max - v_min > 10:
        ticks = np.array([int(np.round(t)) for t in ticks])
        ticks = np.unique(ticks)
    else:
        ticks = np.unique(np.round(ticks, 1))
    
    if ticks.size == 0:  # Fallback
        ticks = np.array([v_min, v_max])
    
    def _get_markersize(size: float) -> float:
        normalized_size = size_norm(size)
        actual_size = min(sizes[0] + normalized_size * (sizes[1] - sizes[0]), sizes[1]) 
        # Convert to markersize for legend
        return np.sqrt(actual_size / np.pi) * 2

    return [str(t) for t in ticks], [_get_markersize(t) for t in ticks]
    
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
        alpha: Optional[float] = None,
        linewidth: Optional[float] = None,
        kwargs: Optional[Dict] = None,
        create_legend: bool = True,
    ) -> None:
    """
    Create legend handles for scatter plot.

    Parameters
    ----------
    create_legend : bool, default=True
        If True, creates the legend immediately. If False, only stores metadata.
    """
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)

    kwargs = kwargs or {}
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="circle")

    # Store legend data in collection for later retrieval
    legend_data = {}

    # Prepare hue legend data
    if hue is not None:
        hue_label = kwargs.pop("hue_label", hue)
        if isinstance(palette, dict):  # categorical legend
            hue_handles = create_legend_handles(
                labels=list(palette.keys()),
                colors=list(palette.values()),
                **handle_kwargs
            )
            legend_data["hue"] = {
                "handles": hue_handles,
                "label": hue_label,
            }
        else:
            mappable = ScalarMappable(norm=hue_norm, cmap=palette)
            legend_data["hue"] = {
                "mappable": mappable,
                "label": hue_label,
                "height": kwargs.pop("hue_height", 20),  # millimeters (was 0.2 axes coords)
                "width": kwargs.pop("hue_width", 5),    # millimeters (was 0.05 axes coords)
                "type": "colorbar",
            }

    # Prepare size legend data
    if size is not None:
        tick_color = color if hue is None else "gray"
        size_handle_kwargs = handle_kwargs.copy()
        size_handle_kwargs["color"] = tick_color
        tick_labels, tick_sizes = _get_size_ticks(
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
        legend_data["size"] = {
            "handles": size_handles,
            "label": kwargs.pop("size_label", size),
            "labelspacing": kwargs.pop("labelspacing", 1/3 * max(1, sizes[1] / 200)),
        }

    # Prepare style legend data
    if style is not None:
        style_values = data[style].unique()
        style_label = kwargs.pop("style_label", style)

        # Use resolve_marker_map to get marker mapping
        # If markers is True (from seaborn default), treat as None for our mapping
        marker_param = markers if isinstance(markers, (list, dict)) else None
        marker_map = resolve_marker_map(
            values=list(style_values),
            marker_map=marker_param
        )

        # Determine color for style legend
        style_color = color if hue is None else "gray"
        style_handle_kwargs = handle_kwargs.copy()
        style_handle_kwargs["color"] = style_color
        style_handle_kwargs.pop("style")  # Remove style key from handle_kwargs

        # Create legend handles with different markers
        style_handles = create_legend_handles(
            labels=[str(val) for val in style_values],
            markers=[marker_map[val] for val in style_values],
            **style_handle_kwargs
        )
        legend_data["style"] = {
            "handles": style_handles,
            "label": style_label,
        }

    # Store metadata on collection (always, regardless of create_legend)
    if len(ax.collections) > 0:
        ax.collections[0]._legend_data = legend_data

    # Create legends using new legend() API (only if create_legend=True)
    if create_legend:
        builder = legend(ax=ax)