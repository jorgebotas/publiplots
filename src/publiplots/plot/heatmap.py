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
    and color encodes another.
    """
    fig = ax.get_figure()
    legend_kws = legend_kws or {}

    # Default sizes
    if sizes is None:
        sizes = (20, 500)

    # Flatten matrices for plotting
    n_rows, n_cols = matrix.shape

    # Create coordinate grids
    x_coords = []
    y_coords = []
    colors = []
    marker_sizes = []

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

            x_coords.append(j)
            y_coords.append(i)
            colors.append(val)
            marker_sizes.append(size_val)

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    colors = np.array(colors)
    marker_sizes = np.array(marker_sizes)

    # Color normalization
    c_min = vmin if vmin is not None else colors.min()
    c_max = vmax if vmax is not None else colors.max()

    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        color_norm = TwoSlopeNorm(vmin=c_min, vcenter=center, vmax=c_max)
    else:
        color_norm = Normalize(vmin=c_min, vmax=c_max)

    # Size normalization
    if size_norm is None:
        size_norm = Normalize(vmin=marker_sizes.min(), vmax=marker_sizes.max())
    elif isinstance(size_norm, tuple):
        size_norm = Normalize(vmin=size_norm[0], vmax=size_norm[1])

    # Map sizes to point sizes
    normalized_sizes = size_norm(marker_sizes)
    point_sizes = sizes[0] + normalized_sizes * (sizes[1] - sizes[0])

    # Draw scatter
    scatter = ax.scatter(
        x_coords + 0.5,  # Center in cells
        y_coords + 0.5,
        c=colors,
        s=point_sizes,
        cmap=cmap,
        norm=color_norm,
        edgecolors=edgecolor if edgecolor else "face",
        linewidths=linewidth,
        zorder=2,
        **kwargs
    )

    # Apply publiplots transparency styling
    apply_transparency(scatter, face_alpha=alpha, edge_alpha=1.0)

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
        size_handles, size_labels = _create_size_legend(
            marker_sizes=marker_sizes,
            sizes=sizes,
            size_norm=size_norm,
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


# =============================================================================
# Complex Heatmap Builder (Stage 2)
# =============================================================================

# Conversion constant: millimeters to inches
MM2INCH = 1 / 25.4


def complex_heatmap(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    value: Optional[str] = None,
    size: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    annot: Union[bool, pd.DataFrame] = False,
    annot_kws: Optional[Dict] = None,
    fmt: str = ".2g",
    linewidths: float = 0,
    linecolor: str = "white",
    square: bool = False,
    sizes: Optional[Tuple[float, float]] = None,
    size_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    xticklabels: Union[bool, str, List] = "auto",
    yticklabels: Union[bool, str, List] = "auto",
    mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    # Clustering parameters
    row_cluster: bool = False,
    col_cluster: bool = False,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
    row_dendrogram: bool = True,
    col_dendrogram: bool = True,
    dendrogram_size: float = 10,
    # Layout parameters
    hspace: float = 1.0,
    wspace: float = 1.0,
    **kwargs
) -> 'ComplexHeatmapBuilder':
    """
    Create a complex heatmap builder for adding margin plots.

    This function returns a builder object that allows adding plots to the
    margins of a heatmap using method chaining. Call `.build()` to render
    the final figure.

    Parameters
    ----------
    data : pd.DataFrame
        Input data (wide or long format).
    x, y, value, size : str, optional
        Column names for long-format data.
    cmap : str, default="viridis"
        Colormap for the heatmap.
    vmin, vmax : float, optional
        Color normalization bounds.
    center : float, optional
        Center value for diverging colormaps.
    annot : bool or DataFrame, default=False
        Cell annotations.
    annot_kws : dict, optional
        Annotation styling.
    fmt : str, default=".2g"
        Annotation format string.
    linewidths : float, default=0
        Cell border width.
    linecolor : str, default="white"
        Cell border color.
    square : bool, default=False
        Force square cells.
    sizes : tuple, optional
        (min, max) marker sizes for dot heatmap.
    size_norm : tuple or Normalize, optional
        Size normalization.
    alpha : float, optional
        Marker transparency.
    linewidth : float, optional
        Marker edge width.
    edgecolor : str, optional
        Marker edge color.
    figsize : tuple, optional
        Base figure size (will be adjusted for margins).
    title : str, default=""
        Plot title.
    xlabel, ylabel : str, default=""
        Axis labels.
    legend : bool, default=True
        Show legend.
    legend_kws : dict, optional
        Legend styling.
    xticklabels, yticklabels : bool, str, or list
        Tick label configuration.
    mask : DataFrame or array, optional
        Cell mask.
    row_cluster : bool, default=False
        Cluster rows.
    col_cluster : bool, default=False
        Cluster columns.
    cluster_method : str, default="ward"
        Clustering linkage method.
    cluster_metric : str, default="euclidean"
        Clustering distance metric.
    row_dendrogram : bool, default=True
        Show row dendrogram if clustering.
    col_dendrogram : bool, default=True
        Show column dendrogram if clustering.
    dendrogram_size : float, default=10
        Dendrogram size in mm.
    hspace : float, default=1.0
        Vertical space between plots (mm).
    wspace : float, default=1.0
        Horizontal space between plots (mm).
    **kwargs
        Additional heatmap parameters.

    Returns
    -------
    ComplexHeatmapBuilder
        Builder object with `.add_top()`, `.add_left()`, etc. methods.

    Examples
    --------
    Basic complex heatmap with margins:

    >>> fig, axes = (
    ...     pp.complex_heatmap(data, x="sample", y="gene", value="expr")
    ...     .add_top(pp.barplot, data=totals, x="sample", y="count", height=15)
    ...     .add_left(pp.barplot, data=means, x="mean", y="gene", width=15)
    ...     .build()
    ... )

    With clustering and dendrograms:

    >>> fig, axes = (
    ...     pp.complex_heatmap(matrix, row_cluster=True, col_cluster=True)
    ...     .build()
    ... )

    Access individual axes:

    >>> axes['main']      # Main heatmap
    >>> axes['top'][0]    # First top margin plot
    >>> axes['left'][0]   # First left margin plot
    """
    return ComplexHeatmapBuilder(
        data=data,
        x=x,
        y=y,
        value=value,
        size=size,
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
        sizes=sizes,
        size_norm=size_norm,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_kws=legend_kws,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        mask=mask,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
        row_dendrogram=row_dendrogram,
        col_dendrogram=col_dendrogram,
        dendrogram_size=dendrogram_size,
        hspace=hspace,
        wspace=wspace,
        **kwargs
    )


class ComplexHeatmapBuilder:
    """
    Builder for complex heatmaps with margin plots.

    This class enables composing heatmaps with additional plots in the margins
    (top, bottom, left, right) using method chaining.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        value: Optional[str] = None,
        size: Optional[str] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None,
        annot: Union[bool, pd.DataFrame] = False,
        annot_kws: Optional[Dict] = None,
        fmt: str = ".2g",
        linewidths: float = 0,
        linecolor: str = "white",
        square: bool = False,
        sizes: Optional[Tuple[float, float]] = None,
        size_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
        alpha: Optional[float] = None,
        linewidth: Optional[float] = None,
        edgecolor: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        legend: bool = True,
        legend_kws: Optional[Dict] = None,
        xticklabels: Union[bool, str, List] = "auto",
        yticklabels: Union[bool, str, List] = "auto",
        mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        row_cluster: bool = False,
        col_cluster: bool = False,
        cluster_method: str = "ward",
        cluster_metric: str = "euclidean",
        row_dendrogram: bool = True,
        col_dendrogram: bool = True,
        dendrogram_size: float = 10,
        hspace: float = 1.0,
        wspace: float = 1.0,
        **kwargs
    ):
        # Store heatmap parameters
        self._heatmap_params = {
            "data": data,
            "x": x,
            "y": y,
            "value": value,
            "size": size,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "center": center,
            "annot": annot,
            "annot_kws": annot_kws,
            "fmt": fmt,
            "linewidths": linewidths,
            "linecolor": linecolor,
            "square": square,
            "sizes": sizes,
            "size_norm": size_norm,
            "alpha": alpha,
            "linewidth": linewidth,
            "edgecolor": edgecolor,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "legend": legend,
            "legend_kws": legend_kws,
            "xticklabels": xticklabels,
            "yticklabels": yticklabels,
            "mask": mask,
        }
        self._heatmap_params.update(kwargs)

        # Layout parameters
        self._figsize = figsize or resolve_param("figure.figsize")
        self._hspace = hspace
        self._wspace = wspace

        # Clustering parameters
        self._row_cluster = row_cluster
        self._col_cluster = col_cluster
        self._cluster_method = cluster_method
        self._cluster_metric = cluster_metric
        self._row_dendrogram = row_dendrogram
        self._col_dendrogram = col_dendrogram
        self._dendrogram_size = dendrogram_size

        # Margin plots storage
        self._margins = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
        }

        # Computed data (set during build)
        self._matrix = None
        self._row_order = None
        self._col_order = None
        self._row_linkage = None
        self._col_linkage = None

    def add_top(
        self,
        func,
        height: float = 15,
        data: Optional[pd.DataFrame] = None,
        align: bool = True,
        gap: float = 0,
        **kwargs
    ) -> 'ComplexHeatmapBuilder':
        """
        Add a plot to the top margin.

        Parameters
        ----------
        func : callable
            Plot function (e.g., pp.barplot). Must accept `ax` parameter.
        height : float, default=15
            Height in millimeters.
        data : DataFrame, optional
            Data for the plot.
        align : bool, default=True
            Align x-axis with heatmap columns.
        gap : float, default=0
            Additional gap from previous element (mm).
        **kwargs
            Additional parameters passed to func.

        Returns
        -------
        self : ComplexHeatmapBuilder
            Returns self for method chaining.
        """
        self._margins['top'].append({
            'func': func,
            'size': height,
            'data': data,
            'align': align,
            'gap': gap,
            'kwargs': kwargs,
        })
        return self

    def add_bottom(
        self,
        func,
        height: float = 15,
        data: Optional[pd.DataFrame] = None,
        align: bool = True,
        gap: float = 0,
        **kwargs
    ) -> 'ComplexHeatmapBuilder':
        """
        Add a plot to the bottom margin.

        Parameters
        ----------
        func : callable
            Plot function. Must accept `ax` parameter.
        height : float, default=15
            Height in millimeters.
        data : DataFrame, optional
            Data for the plot.
        align : bool, default=True
            Align x-axis with heatmap columns.
        gap : float, default=0
            Additional gap from previous element (mm).
        **kwargs
            Additional parameters passed to func.

        Returns
        -------
        self : ComplexHeatmapBuilder
        """
        self._margins['bottom'].append({
            'func': func,
            'size': height,
            'data': data,
            'align': align,
            'gap': gap,
            'kwargs': kwargs,
        })
        return self

    def add_left(
        self,
        func,
        width: float = 15,
        data: Optional[pd.DataFrame] = None,
        align: bool = True,
        gap: float = 0,
        **kwargs
    ) -> 'ComplexHeatmapBuilder':
        """
        Add a plot to the left margin.

        Parameters
        ----------
        func : callable
            Plot function. Must accept `ax` parameter.
        width : float, default=15
            Width in millimeters.
        data : DataFrame, optional
            Data for the plot.
        align : bool, default=True
            Align y-axis with heatmap rows.
        gap : float, default=0
            Additional gap from previous element (mm).
        **kwargs
            Additional parameters passed to func.

        Returns
        -------
        self : ComplexHeatmapBuilder
        """
        self._margins['left'].append({
            'func': func,
            'size': width,
            'data': data,
            'align': align,
            'gap': gap,
            'kwargs': kwargs,
        })
        return self

    def add_right(
        self,
        func,
        width: float = 15,
        data: Optional[pd.DataFrame] = None,
        align: bool = True,
        gap: float = 0,
        **kwargs
    ) -> 'ComplexHeatmapBuilder':
        """
        Add a plot to the right margin.

        Parameters
        ----------
        func : callable
            Plot function. Must accept `ax` parameter.
        width : float, default=15
            Width in millimeters.
        data : DataFrame, optional
            Data for the plot.
        align : bool, default=True
            Align y-axis with heatmap rows.
        gap : float, default=0
            Additional gap from previous element (mm).
        **kwargs
            Additional parameters passed to func.

        Returns
        -------
        self : ComplexHeatmapBuilder
        """
        self._margins['right'].append({
            'func': func,
            'size': width,
            'data': data,
            'align': align,
            'gap': gap,
            'kwargs': kwargs,
        })
        return self

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare and optionally cluster the heatmap data."""
        data = self._heatmap_params['data']
        x = self._heatmap_params['x']
        y = self._heatmap_params['y']
        value = self._heatmap_params['value']

        # Convert to matrix if long format
        if x is not None and y is not None and value is not None:
            matrix = data.pivot(index=y, columns=x, values=value)
        else:
            matrix = data.copy()

        # Apply clustering if requested
        if self._row_cluster or self._col_cluster:
            matrix, self._row_order, self._col_order, self._row_linkage, self._col_linkage = \
                _cluster_data(
                    matrix,
                    row_cluster=self._row_cluster,
                    col_cluster=self._col_cluster,
                    method=self._cluster_method,
                    metric=self._cluster_metric,
                )

        self._matrix = matrix
        return matrix

    def _add_clustering_dendrograms(self):
        """Add dendrogram margin plots if clustering is enabled."""
        if self._col_cluster and self._col_dendrogram and self._col_linkage is not None:
            # Add column dendrogram to top
            # Don't align dendrograms - they have their own coordinate system
            self._margins['top'].insert(0, {
                'func': dendrogram,
                'size': self._dendrogram_size,
                'data': None,
                'align': False,  # Dendrograms don't share axes
                'gap': 0,
                'kwargs': {
                    'linkage': self._col_linkage,
                    'orientation': 'top',
                },
            })

        if self._row_cluster and self._row_dendrogram and self._row_linkage is not None:
            # Add row dendrogram to left
            # Don't align dendrograms - they have their own coordinate system
            self._margins['left'].insert(0, {
                'func': dendrogram,
                'size': self._dendrogram_size,
                'data': None,
                'align': False,  # Dendrograms don't share axes
                'gap': 0,
                'kwargs': {
                    'linkage': self._row_linkage,
                    'orientation': 'left',
                },
            })

    def build(self) -> Tuple[plt.Figure, Dict[str, Union[Axes, List[Axes]]]]:
        """
        Build the complex heatmap with all margin plots.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        axes : dict
            Dictionary with keys 'main', 'top', 'bottom', 'left', 'right'.
            Each margin key contains a list of axes.
        """
        from matplotlib import gridspec

        # Prepare data (with optional clustering)
        matrix = self._prepare_data()

        # Add dendrograms if clustering enabled
        self._add_clustering_dendrograms()

        # Count margin plots
        n_top = len(self._margins['top'])
        n_bottom = len(self._margins['bottom'])
        n_left = len(self._margins['left'])
        n_right = len(self._margins['right'])

        # Calculate grid dimensions
        n_rows = n_top + 1 + n_bottom
        n_cols = n_left + 1 + n_right

        # Calculate size ratios
        # Convert mm sizes to relative ratios
        main_height = self._figsize[1]  # inches
        main_width = self._figsize[0]   # inches

        # Top margins (in reverse order - first added is closest to heatmap)
        top_sizes = [m['size'] * MM2INCH for m in reversed(self._margins['top'])]
        bottom_sizes = [m['size'] * MM2INCH for m in self._margins['bottom']]
        left_sizes = [m['size'] * MM2INCH for m in reversed(self._margins['left'])]
        right_sizes = [m['size'] * MM2INCH for m in self._margins['right']]

        # Build height and width ratios
        height_ratios = top_sizes + [main_height] + bottom_sizes
        width_ratios = left_sizes + [main_width] + right_sizes

        # Calculate total figure size
        total_height = sum(height_ratios) + (n_rows - 1) * self._hspace * MM2INCH
        total_width = sum(width_ratios) + (n_cols - 1) * self._wspace * MM2INCH

        # Create figure
        fig = plt.figure(figsize=(total_width, total_height))

        # Create GridSpec
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=self._hspace * MM2INCH / main_height,
            wspace=self._wspace * MM2INCH / main_width,
        )

        # Create main heatmap axes
        main_row = n_top
        main_col = n_left
        ax_main = fig.add_subplot(gs[main_row, main_col])

        # Prepare heatmap params for the clustered data
        heatmap_params = self._heatmap_params.copy()
        if self._row_cluster or self._col_cluster:
            # Use the clustered matrix directly (wide format)
            heatmap_params['data'] = self._matrix
            heatmap_params['x'] = None
            heatmap_params['y'] = None
            heatmap_params['value'] = None

        # Draw main heatmap
        heatmap(ax=ax_main, **heatmap_params)

        # Initialize axes dict
        axes = {
            'main': ax_main,
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
        }

        # Draw top margins (from closest to heatmap outward)
        for i, margin in enumerate(reversed(self._margins['top'])):
            row_idx = n_top - 1 - i
            ax = fig.add_subplot(
                gs[row_idx, main_col],
                sharex=ax_main if margin['align'] else None
            )
            self._draw_margin_plot(ax, margin, 'top')
            if margin['align']:
                ax.tick_params(labelbottom=False)
                plt.setp(ax.get_xticklabels(), visible=False)
            axes['top'].append(ax)

        # Draw bottom margins
        for i, margin in enumerate(self._margins['bottom']):
            row_idx = n_top + 1 + i
            ax = fig.add_subplot(
                gs[row_idx, main_col],
                sharex=ax_main if margin['align'] else None
            )
            self._draw_margin_plot(ax, margin, 'bottom')
            if margin['align'] and i < len(self._margins['bottom']) - 1:
                ax.tick_params(labeltop=False)
            axes['bottom'].append(ax)

        # Draw left margins (from closest to heatmap outward)
        for i, margin in enumerate(reversed(self._margins['left'])):
            col_idx = n_left - 1 - i
            ax = fig.add_subplot(
                gs[main_row, col_idx],
                sharey=ax_main if margin['align'] else None
            )
            self._draw_margin_plot(ax, margin, 'left')
            if margin['align']:
                ax.tick_params(labelright=False)
                plt.setp(ax.get_yticklabels(), visible=False)
            axes['left'].append(ax)

        # Draw right margins
        for i, margin in enumerate(self._margins['right']):
            col_idx = n_left + 1 + i
            ax = fig.add_subplot(
                gs[main_row, col_idx],
                sharey=ax_main if margin['align'] else None
            )
            self._draw_margin_plot(ax, margin, 'right')
            if margin['align'] and i < len(self._margins['right']) - 1:
                ax.tick_params(labelleft=False)
            axes['right'].append(ax)

        # Hide main heatmap tick labels if there are aligned margin plots
        if any(m['align'] for m in self._margins['top']):
            ax_main.tick_params(labeltop=False)
        if any(m['align'] for m in self._margins['left']):
            ax_main.tick_params(labelleft=False)

        return fig, axes

    def _draw_margin_plot(self, ax: Axes, margin: Dict, position: str):
        """Draw a single margin plot."""
        func = margin['func']
        data = margin['data']
        kwargs = margin['kwargs'].copy()

        # Add ax to kwargs
        kwargs['ax'] = ax

        # Add data if provided
        if data is not None:
            kwargs['data'] = data

        # Disable legend by default for margin plots
        if 'legend' not in kwargs:
            kwargs['legend'] = False

        # Call the plot function
        try:
            func(**kwargs)
        except TypeError:
            # Some functions may not accept all kwargs
            # Try with just essential params
            essential = {'ax': ax}
            if data is not None:
                essential['data'] = data
            for key in ['x', 'y', 'linkage', 'orientation']:
                if key in kwargs:
                    essential[key] = kwargs[key]
            func(**essential)


def _cluster_data(
    matrix: pd.DataFrame,
    row_cluster: bool = True,
    col_cluster: bool = True,
    method: str = "ward",
    metric: str = "euclidean",
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Cluster rows and/or columns of a matrix.

    Returns
    -------
    matrix : DataFrame
        Reordered matrix.
    row_order : ndarray or None
        Row ordering indices.
    col_order : ndarray or None
        Column ordering indices.
    row_linkage : ndarray or None
        Row linkage matrix.
    col_linkage : ndarray or None
        Column linkage matrix.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    row_order = None
    col_order = None
    row_linkage = None
    col_linkage = None

    # Cluster rows
    if row_cluster:
        # Compute linkage
        row_distances = pdist(matrix.values, metric=metric)
        row_linkage = linkage(row_distances, method=method)
        row_order = leaves_list(row_linkage)
        matrix = matrix.iloc[row_order, :]

    # Cluster columns
    if col_cluster:
        # Compute linkage
        col_distances = pdist(matrix.values.T, metric=metric)
        col_linkage = linkage(col_distances, method=method)
        col_order = leaves_list(col_linkage)
        matrix = matrix.iloc[:, col_order]

    return matrix, row_order, col_order, row_linkage, col_linkage


def dendrogram(
    linkage: Optional[np.ndarray] = None,
    data: Optional[pd.DataFrame] = None,
    method: str = "ward",
    metric: str = "euclidean",
    orientation: str = "top",
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Draw a dendrogram.

    Parameters
    ----------
    linkage : ndarray, optional
        Precomputed linkage matrix. If None, computed from data.
    data : DataFrame, optional
        Data to cluster (if linkage not provided).
    method : str, default="ward"
        Clustering linkage method.
    metric : str, default="euclidean"
        Distance metric.
    orientation : str, default="top"
        Dendrogram orientation: 'top', 'bottom', 'left', 'right'.
    color : str, optional
        Line color. Uses rcParams default if None.
    linewidth : float, optional
        Line width. Uses rcParams default if None.
    ax : Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments passed to scipy.dendrogram.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    from scipy.cluster.hierarchy import linkage as compute_linkage
    from scipy.spatial.distance import pdist

    # Resolve defaults
    color = resolve_param("color", color)
    linewidth = resolve_param("lines.linewidth", linewidth)

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Compute linkage if not provided
    if linkage is None:
        if data is None:
            raise ValueError("Either linkage or data must be provided")
        distances = pdist(data.values, metric=metric)
        linkage = compute_linkage(distances, method=method)

    # Draw dendrogram
    scipy_dendrogram(
        linkage,
        ax=ax,
        orientation=orientation,
        no_labels=True,
        color_threshold=0,
        above_threshold_color=color,
        **kwargs
    )

    # Style the dendrogram lines
    for line in ax.collections:
        line.set_linewidth(linewidth)
    for line in ax.lines:
        line.set_linewidth(linewidth)
        line.set_color(color)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adjust axis limits based on orientation
    if orientation in ['top', 'bottom']:
        ax.set_xlim(ax.get_xlim())
    else:
        ax.set_ylim(ax.get_ylim())

    return fig, ax
