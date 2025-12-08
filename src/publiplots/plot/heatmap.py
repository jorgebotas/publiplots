"""
Heatmap visualization module for publiplots.

Provides flexible heatmap visualizations with support for both wide-format
(matrix) and long-format (tidy) data, color encoding, size encoding for
dot/bubble heatmaps, and publication-ready styling.
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
from publiplots.utils import is_categorical, is_numeric, create_legend_handles
from publiplots.utils.legend import legend as legend_builder
from publiplots.utils.transparency import apply_transparency
from publiplots.plot.scatter import scatterplot


def heatmap(
    data: pd.DataFrame,
    # Long-format parameters (optional)
    x: Optional[str] = None,
    y: Optional[str] = None,
    value: Optional[str] = None,
    # Dot heatmap mode
    size: Optional[str] = None,
    # Color encoding
    cmap: Optional[str] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    # Annotations
    annot: Union[bool, pd.DataFrame] = False,
    annot_kws: Optional[Dict] = None,
    fmt: str = ".2g",
    # Styling
    linewidths: float = 0,
    linecolor: str = "white",
    square: bool = False,
    # Size encoding (dot mode)
    sizes: Optional[Tuple[float, float]] = None,
    size_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    # Standard publiplots params
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    # Legend
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    # Additional options
    xticklabels: Union[bool, str, List] = "auto",
    yticklabels: Union[bool, str, List] = "auto",
    mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create a heatmap with publiplots styling.

    This function creates heatmaps for both wide-format (matrix) and long-format
    (tidy) data. When `size` is provided, creates a dot/bubble heatmap where
    marker size encodes an additional variable.

    Parameters
    ----------
    data : pd.DataFrame
        Input data. Can be:
        - Wide format: DataFrame where index=rows, columns=cols, values=cells
        - Long format: DataFrame with x, y, value columns (requires x, y, value params)
    x : str, optional
        Column name for x-axis (columns) in long-format data.
    y : str, optional
        Column name for y-axis (rows) in long-format data.
    value : str, optional
        Column name for cell values in long-format data.
    size : str, optional
        Column name for marker sizes (long-format only). If provided, creates
        a dot/bubble heatmap instead of a color heatmap.
    cmap : str or colormap, default="viridis"
        Colormap for mapping values to colors.
    vmin : float, optional
        Minimum value for color normalization. If None, uses data minimum.
    vmax : float, optional
        Maximum value for color normalization. If None, uses data maximum.
    center : float, optional
        Value to center the colormap at (e.g., 0 for diverging data).
    annot : bool or DataFrame, default=False
        If True, write data values in cells. If DataFrame, use its values.
    annot_kws : dict, optional
        Keyword arguments for annotation text.
    fmt : str, default=".2g"
        Format string for annotations.
    linewidths : float, default=0
        Width of lines between cells.
    linecolor : str, default="white"
        Color of lines between cells.
    square : bool, default=False
        If True, set aspect ratio to equal for square cells.
    sizes : tuple of float, optional
        (min_size, max_size) in points^2 for dot heatmap markers.
        Default: (20, 500).
    size_norm : tuple or Normalize, optional
        Normalization for size values. If tuple, (vmin, vmax).
    alpha : float, optional
        Transparency for markers in dot mode. Uses rcParams default.
    linewidth : float, optional
        Edge linewidth for markers in dot mode. Uses rcParams default.
    edgecolor : str, optional
        Edge color for markers in dot mode. If None, uses marker color.
    figsize : tuple, optional
        Figure size (width, height). Uses rcParams default.
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label.
    ylabel : str, default=""
        Y-axis label.
    legend : bool, default=True
        Whether to show colorbar/size legend.
    legend_kws : dict, optional
        Keyword arguments for legend:
        - value_label : str - Label for colorbar (default: value column name)
        - size_label : str - Label for size legend (default: size column name)
    xticklabels : bool, str, or list, default="auto"
        X-axis tick labels.
    yticklabels : bool, str, or list, default="auto"
        Y-axis tick labels.
    mask : DataFrame or array, optional
        Boolean mask for hiding cells (True = hidden).
    **kwargs
        Additional keyword arguments passed to sns.heatmap (standard mode)
        or internal scatter function (dot mode).

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Wide-format heatmap:

    >>> matrix = pd.DataFrame(np.random.randn(10, 10))
    >>> fig, ax = pp.heatmap(matrix, cmap="coolwarm", center=0)

    Long-format heatmap:

    >>> fig, ax = pp.heatmap(df, x="sample", y="gene", value="expression")

    Dot heatmap with size encoding:

    >>> fig, ax = pp.heatmap(df, x="sample", y="gene",
    ...                       value="expression", size="pvalue")

    Annotated heatmap:

    >>> fig, ax = pp.heatmap(matrix, annot=True, fmt=".1f")
    """
    # Read defaults from rcParams if not provided
    figsize = resolve_param("figure.figsize", figsize)
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine data format and convert if needed
    is_long_format = x is not None and y is not None and value is not None

    if is_long_format:
        # Long format -> pivot to wide
        matrix = data.pivot(index=y, columns=x, values=value)
        size_matrix = data.pivot(index=y, columns=x, values=size) if size else None

        # Store original labels
        x_labels = matrix.columns.tolist()
        y_labels = matrix.index.tolist()
    else:
        # Wide format - use as-is
        matrix = data.copy()
        size_matrix = None
        x_labels = matrix.columns.tolist()
        y_labels = matrix.index.tolist()

        # In wide format, size must be None (not supported)
        if size is not None:
            raise ValueError(
                "size parameter requires long-format data. "
                "Provide x, y, and value parameters for long-format."
            )

    # Handle mask for long-format data
    if mask is not None and is_long_format:
        # Pivot mask if needed
        if isinstance(mask, pd.DataFrame) and mask.shape != matrix.shape:
            # Assume mask needs pivoting
            mask = None  # Skip mask handling for now

    # Determine mode: dot heatmap vs standard heatmap
    if size is not None and size_matrix is not None:
        # DOT HEATMAP MODE
        fig, ax = _draw_dot_heatmap(
            ax=ax,
            matrix=matrix,
            size_matrix=size_matrix,
            x_labels=x_labels,
            y_labels=y_labels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            sizes=sizes,
            size_norm=size_norm,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor=edgecolor,
            square=square,
            mask=mask,
            legend=legend,
            legend_kws=legend_kws,
            value_col=value,
            size_col=size,
            **kwargs
        )
    else:
        # STANDARD HEATMAP MODE
        fig, ax = _draw_heatmap(
            ax=ax,
            matrix=matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=annot,
            annot_kws=annot_kws,
            fmt=fmt,
            linewidths=linewidths,
            linecolor=linecolor,
            square=square,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            mask=mask,
            legend=legend,
            legend_kws=legend_kws,
            value_col=value,
            **kwargs
        )

    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax


def _draw_heatmap(
    ax: Axes,
    matrix: pd.DataFrame,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    center: Optional[float],
    annot: Union[bool, pd.DataFrame],
    annot_kws: Optional[Dict],
    fmt: str,
    linewidths: float,
    linecolor: str,
    square: bool,
    xticklabels: Union[bool, str, List],
    yticklabels: Union[bool, str, List],
    mask: Optional[Union[pd.DataFrame, np.ndarray]],
    legend: bool,
    legend_kws: Optional[Dict],
    value_col: Optional[str],
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Draw standard color-encoded heatmap using seaborn.
    """
    fig = ax.get_figure()
    legend_kws = legend_kws or {}

    # Prepare seaborn kwargs
    heatmap_kwargs = {
        "data": matrix,
        "ax": ax,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "center": center,
        "annot": annot,
        "fmt": fmt,
        "linewidths": linewidths,
        "linecolor": linecolor,
        "square": square,
        "xticklabels": xticklabels,
        "yticklabels": yticklabels,
        "mask": mask,
        "cbar": False,  # We'll add our own colorbar via legend system
    }

    if annot_kws is not None:
        heatmap_kwargs["annot_kws"] = annot_kws

    # Merge with user kwargs
    heatmap_kwargs.update(kwargs)

    # Draw heatmap
    sns.heatmap(**heatmap_kwargs)

    # Add colorbar via legend system
    if legend:
        # Determine normalization
        v_min = vmin if vmin is not None else matrix.min().min()
        v_max = vmax if vmax is not None else matrix.max().max()

        if center is not None:
            # Diverging normalization
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=v_min, vcenter=center, vmax=v_max)
        else:
            norm = Normalize(vmin=v_min, vmax=v_max)

        mappable = ScalarMappable(norm=norm, cmap=cmap)

        builder = legend_builder(ax=ax, auto=False)
        builder.add_colorbar(
            mappable=mappable,
            label=legend_kws.get("value_label", value_col or ""),
            height=legend_kws.get("height", 20),
            width=legend_kws.get("width", 5),
        )

    return fig, ax


def _draw_dot_heatmap(
    ax: Axes,
    matrix: pd.DataFrame,
    size_matrix: pd.DataFrame,
    x_labels: List,
    y_labels: List,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    center: Optional[float],
    sizes: Optional[Tuple[float, float]],
    size_norm: Optional[Union[Tuple[float, float], Normalize]],
    alpha: float,
    linewidth: float,
    edgecolor: Optional[str],
    square: bool,
    mask: Optional[Union[pd.DataFrame, np.ndarray]],
    legend: bool,
    legend_kws: Optional[Dict],
    value_col: str,
    size_col: str,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Draw dot/bubble heatmap where marker size encodes one variable
    and color encodes another. Uses pp.scatterplot for modularity.
    """
    fig = ax.get_figure()
    legend_kws = legend_kws or {}

    # Default sizes
    if sizes is None:
        sizes = (20, 500)

    # Flatten matrices for plotting - create long-format data
    n_rows, n_cols = matrix.shape

    plot_data = []
    for i, row_label in enumerate(y_labels):
        for j, col_label in enumerate(x_labels):
            # Check mask
            if mask is not None:
                if isinstance(mask, pd.DataFrame):
                    if mask.iloc[i, j]:
                        continue
                elif mask[i, j]:
                    continue

            val = matrix.iloc[i, j]
            size_val = size_matrix.iloc[i, j]

            # Skip NaN values
            if pd.isna(val) or pd.isna(size_val):
                continue

            plot_data.append({
                'x': j + 0.5,  # Center in cells
                'y': i + 0.5,
                'value': val,
                'size_value': size_val,
            })

    # Convert to DataFrame for pp.scatterplot
    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        # Empty plot
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        return fig, ax

    # Prepare hue_norm for continuous color mapping
    c_min = vmin if vmin is not None else plot_df['value'].min()
    c_max = vmax if vmax is not None else plot_df['value'].max()

    if center is not None:
        # For centered colormaps, we need to use a custom norm
        # pp.scatterplot doesn't directly support TwoSlopeNorm via hue_norm
        # So we'll handle this after calling scatterplot
        hue_norm_param = (c_min, c_max)
    else:
        hue_norm_param = (c_min, c_max)

    # Call pp.scatterplot for modular scatter rendering
    scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='value',
        size='size_value',
        palette=cmap,  # Use cmap as palette for continuous color
        sizes=sizes,
        size_norm=size_norm,
        hue_norm=hue_norm_param,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        ax=ax,
        legend=False,  # We'll add custom legend below
        margins=0,  # No extra margins for heatmap
        **kwargs
    )

    # If center is specified, we need to update the color normalization
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        color_norm = TwoSlopeNorm(vmin=c_min, vcenter=center, vmax=c_max)

        # Update the collection colors with centered normalization
        for collection in ax.collections:
            # Recompute colors with centered norm
            colors_array = plot_df['value'].values
            collection.set_array(colors_array)
            collection.set_norm(color_norm)
            collection.set_cmap(cmap)
    else:
        color_norm = Normalize(vmin=c_min, vmax=c_max)

    # Set axis limits and ticks
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Invert y-axis to match heatmap convention (top to bottom)
    ax.invert_yaxis()

    # Square aspect ratio if requested
    if square:
        ax.set_aspect("equal")

    # Add grid lines between cells
    ax.set_xticks(np.arange(n_cols + 1), minor=True)
    ax.set_yticks(np.arange(n_rows + 1), minor=True)
    ax.grid(which="minor", color="#e0e0e0", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add legends
    if legend:
        builder = legend_builder(ax=ax, auto=False)

        # Color legend (colorbar)
        mappable = ScalarMappable(norm=color_norm, cmap=cmap)
        builder.add_colorbar(
            mappable=mappable,
            label=legend_kws.get("value_label", value_col or ""),
            height=legend_kws.get("colorbar_height", 20),
            width=legend_kws.get("colorbar_width", 5),
        )

        # Size legend
        # Get size normalization from plot
        if size_norm is None:
            size_norm_obj = Normalize(
                vmin=plot_df['size_value'].min(),
                vmax=plot_df['size_value'].max()
            )
        elif isinstance(size_norm, tuple):
            size_norm_obj = Normalize(vmin=size_norm[0], vmax=size_norm[1])
        else:
            size_norm_obj = size_norm

        size_handles, size_labels = _create_size_legend(
            marker_sizes=plot_df['size_value'].values,
            sizes=sizes,
            size_norm=size_norm_obj,
            alpha=alpha,
            linewidth=linewidth,
        )
        builder.add_legend(
            handles=size_handles,
            label=legend_kws.get("size_label", size_col or ""),
        )

    return fig, ax


def _create_size_legend(
    marker_sizes: np.ndarray,
    sizes: Tuple[float, float],
    size_norm: Normalize,
    alpha: float,
    linewidth: float,
    nbins: int = 4,
) -> Tuple[List, List[str]]:
    """
    Create legend handles for size encoding.
    """
    # Get representative size values
    v_min, v_max = size_norm.vmin, size_norm.vmax
    unique_vals = np.unique(marker_sizes[~np.isnan(marker_sizes)])

    if len(unique_vals) <= 4:
        ticks = unique_vals
    else:
        locator = MaxNLocator(nbins=nbins, min_n_ticks=3)
        ticks = locator.tick_values(v_min, v_max)
        ticks = ticks[(ticks >= v_min) & (ticks <= v_max)]

    # Round ticks
    if v_max - v_min > 10:
        ticks = np.array([int(np.round(t)) for t in ticks])
        ticks = np.unique(ticks)
    else:
        ticks = np.unique(np.round(ticks, 1))

    if len(ticks) == 0:
        ticks = np.array([v_min, v_max])

    # Calculate marker sizes for legend
    def get_markersize(val):
        normalized = size_norm(val)
        point_size = sizes[0] + normalized * (sizes[1] - sizes[0])
        return np.sqrt(point_size / np.pi) * 2

    tick_labels = [str(t) for t in ticks]
    tick_sizes = [get_markersize(t) for t in ticks]

    # Create handles using publiplots legend system
    handles = create_legend_handles(
        labels=tick_labels,
        sizes=tick_sizes,
        color="gray",
        alpha=alpha,
        linewidth=linewidth,
        style="circle",
    )

    return handles, tick_labels

