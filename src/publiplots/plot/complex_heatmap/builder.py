"""
Builder class for creating complex heatmaps with margin plots.
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, List

from publiplots.themes.rcparams import resolve_param
from publiplots.utils.offset import offset_patches, offset_lines, offset_collections
from publiplots.utils.text import calculate_label_space
from .dendrogram import cluster_data, dendrogram as dendrogram_plot, ticklabels as ticklabels_plot

# Conversion constants
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
    xticklabels_kws: Optional[Dict] = None,
    yticklabels_kws: Optional[Dict] = None,
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
    xticklabels_kws : dict, optional
        Keyword arguments for x-axis tick labels. Supports:
        - side: 'top', 'bottom', or 'auto' (default: 'auto')
        - rotation: angle in degrees (default: 0)
        - fontsize: font size in points
        - color: text color
        - va/verticalalignment: vertical alignment
        - ha/horizontalalignment: horizontal alignment
        - and other matplotlib text properties
    yticklabels_kws : dict, optional
        Keyword arguments for y-axis tick labels. Same options as xticklabels_kws,
        but side options are 'left', 'right', or 'auto'.
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
        xticklabels_kws=xticklabels_kws,
        yticklabels_kws=yticklabels_kws,
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
        xticklabels_kws: Optional[Dict] = None,
        yticklabels_kws: Optional[Dict] = None,
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
        hspace: float = 2.0,
        wspace: float = 2.0,
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

        # Store tick label kwargs (with defaults)
        self._xticklabels_kws = xticklabels_kws or {}
        self._yticklabels_kws = yticklabels_kws or {}

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

    def _calculate_label_space_for_position(
        self,
        labels: List[str],
        position: str = 'left',
        fig: Optional[plt.Figure] = None
    ) -> float:
        """
        Calculate space needed for tick labels using actual text measurement.

        This uses matplotlib's text rendering to get accurate measurements,
        matching the approach used in upsetplot for consistent results.

        Note: We pass tick_pad=0 because we draw labels in dedicated axes,
        not using matplotlib's tick system.

        Parameters
        ----------
        labels : list of str
            The tick labels to measure.
        position : str
            Position where labels will be placed: 'left', 'right', 'top', 'bottom'.
        fig : Figure, optional
            Figure to use for measurement. If provided, ensures consistent DPI.

        Returns
        -------
        float
            Required space in inches.
        """
        if not labels:
            return 0.0

        return calculate_label_space(
            labels=labels,
            fig=fig,
            position=position,
            tick_pad=0,  # No tick padding - we draw in dedicated axes, not using tick system
            unit="inches",
            safety_factor=1.0,
        )

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
                cluster_data(
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
            self._margins['top'].insert(0, {
                'func': dendrogram_plot,
                'size': self._dendrogram_size,
                'data': None,
                'align': False,  # Dendrograms have their own coordinate system
                'gap': 0,
                'kwargs': {
                    'linkage': self._col_linkage,
                    'orientation': 'top',
                },
            })

        if self._row_cluster and self._row_dendrogram and self._row_linkage is not None:
            # Add row dendrogram to left
            self._margins['left'].insert(0, {
                'func': dendrogram_plot,
                'size': self._dendrogram_size,
                'data': None,
                'align': False,  # Dendrograms have their own coordinate system
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
        # Import heatmap here to avoid circular imports
        from publiplots.plot.heatmap import heatmap

        # Prepare data (with optional clustering)
        matrix = self._prepare_data()

        # Add dendrograms if clustering enabled
        self._add_clustering_dendrograms()

        # Process xticklabels_kws and yticklabels_kws
        # Extract side and rotation, normalize alignment kwargs
        xticklabels_kws = self._xticklabels_kws.copy()
        yticklabels_kws = self._yticklabels_kws.copy()

        # Pop 'side' from kwargs (with defaults)
        xlabel_side = xticklabels_kws.pop('side', 'auto')
        ylabel_side = yticklabels_kws.pop('side', 'auto')

        # Pop 'rotation' from kwargs (will be passed separately to text)
        xlabel_rotation = xticklabels_kws.pop('rotation', 0)
        ylabel_rotation = yticklabels_kws.pop('rotation', 0)

        # Normalize alignment kwargs (va/verticalalignment, ha/horizontalalignment)
        # Pop them so we can set defaults based on position
        xlabel_va = xticklabels_kws.pop('va', xticklabels_kws.pop('verticalalignment', None))
        xlabel_ha = xticklabels_kws.pop('ha', xticklabels_kws.pop('horizontalalignment', None))
        ylabel_va = yticklabels_kws.pop('va', yticklabels_kws.pop('verticalalignment', None))
        ylabel_ha = yticklabels_kws.pop('ha', yticklabels_kws.pop('horizontalalignment', None))

        # Auto-detect label sides (None means don't add labels automatically)
        if ylabel_side == 'auto':
            # If there are left margins, put ylabel on right
            ylabel_side = 'right' if self._margins['left'] else 'left'

        if xlabel_side == 'auto':
            # If there are top margins, put xlabel on bottom
            xlabel_side = 'bottom' if self._margins['top'] else 'top'

        # Get labels from the matrix
        row_labels = [str(label) for label in matrix.index]
        col_labels = [str(label) for label in matrix.columns]

        # Calculate required space for labels (in inches) only if labels will be added
        ylabel_space = 0.0
        xlabel_space = 0.0
        if ylabel_side is not None or xlabel_side is not None:
            # Create a temporary figure for accurate text measurement
            temp_fig = plt.figure(figsize=self._figsize)

            if ylabel_side is not None:
                ylabel_space = self._calculate_label_space_for_position(row_labels, ylabel_side, fig=temp_fig)
            if xlabel_side is not None:
                xlabel_space = self._calculate_label_space_for_position(col_labels, xlabel_side, fig=temp_fig)

            # Close temp figure - we'll create the real one with proper size
            plt.close(temp_fig)

        # Build text kwargs for labels with position-appropriate defaults
        ylabel_text_kws = {
            'rotation': ylabel_rotation,
            'va': ylabel_va if ylabel_va is not None else 'center',
            'ha': ylabel_ha if ylabel_ha is not None else ('left' if ylabel_side == 'right' else 'right'),
            **yticklabels_kws,  # Remaining kwargs (color, fontsize, etc.)
        }
        xlabel_text_kws = {
            'rotation': xlabel_rotation,
            'va': xlabel_va if xlabel_va is not None else ('top' if xlabel_side == 'bottom' else 'bottom'),
            'ha': xlabel_ha if xlabel_ha is not None else 'center',
            **xticklabels_kws,  # Remaining kwargs (color, fontsize, etc.)
        }

        # Add label layers as stackable margin "plots"
        # Labels should be CLOSEST to heatmap (append to end, not insert at 0)
        # When margins are reversed for drawing, the last item is closest to heatmap
        if ylabel_side == 'right' and row_labels:
            self._margins['right'].append({
                'func': 'labels',
                'size': ylabel_space / MM2INCH,  # Convert back to mm for consistency
                'data': row_labels,
                'align': True,
                'gap': 0,
                'kwargs': {'orientation': 'vertical', 'text_kws': ylabel_text_kws},
            })
        elif ylabel_side == 'left' and row_labels:
            self._margins['left'].append({
                'func': 'labels',
                'size': ylabel_space / MM2INCH,
                'data': row_labels,
                'align': True,
                'gap': 0,
                'kwargs': {'orientation': 'vertical', 'text_kws': ylabel_text_kws},
            })

        if xlabel_side == 'bottom' and col_labels:
            self._margins['bottom'].append({
                'func': 'labels',
                'size': xlabel_space / MM2INCH,
                'data': col_labels,
                'align': True,
                'gap': 0,
                'kwargs': {'orientation': 'horizontal', 'text_kws': xlabel_text_kws},
            })
        elif xlabel_side == 'top' and col_labels:
            self._margins['top'].append({
                'func': 'labels',
                'size': xlabel_space / MM2INCH,
                'data': col_labels,
                'align': True,
                'gap': 0,
                'kwargs': {'orientation': 'horizontal', 'text_kws': xlabel_text_kws},
            })

        # Count margin plots (now includes labels)
        n_top = len(self._margins['top'])
        n_bottom = len(self._margins['bottom'])
        n_left = len(self._margins['left'])
        n_right = len(self._margins['right'])

        # Calculate grid dimensions - no special label columns needed
        n_rows = n_top + 1 + n_bottom
        n_cols = n_left + 1 + n_right

        # Calculate size ratios
        main_height = self._figsize[1]  # inches
        main_width = self._figsize[0]   # inches

        # Convert spacing to inches
        spacing_h = self._hspace * MM2INCH
        spacing_w = self._wspace * MM2INCH

        # Margin sizes - order must match how columns/rows are assigned during drawing
        # Drawing uses reversed iteration with col_idx = n_left - 1 - i, which means:
        #   - First margin in list (e.g., dendrogram) → gets col_idx 0
        #   - Last margin in list (e.g., labels) → gets col_idx n_left-1
        # So sizes should be in ORIGINAL order (not reversed) to match column assignment
        top_sizes = [m['size'] * MM2INCH for m in self._margins['top']]
        bottom_sizes = [m['size'] * MM2INCH for m in self._margins['bottom']]
        left_sizes = [m['size'] * MM2INCH for m in self._margins['left']]
        right_sizes = [m['size'] * MM2INCH for m in self._margins['right']]

        # Build height and width ratios
        height_ratios = top_sizes + [main_height] + bottom_sizes
        width_ratios = left_sizes + [main_width] + right_sizes

        # Calculate GridSpec hspace/wspace as fractions
        # hspace = spacing / average_row_height
        # wspace = spacing / average_col_width
        avg_row_height = sum(height_ratios) / len(height_ratios) if height_ratios else 1
        avg_col_width = sum(width_ratios) / len(width_ratios) if width_ratios else 1
        gs_hspace = spacing_h / avg_row_height if avg_row_height > 0 else 0
        gs_wspace = spacing_w / avg_col_width if avg_col_width > 0 else 0

        # Calculate total figure size including spacing
        # Total gaps = (n_rows - 1) * spacing_h for height, (n_cols - 1) * spacing_w for width
        total_h_spacing = (n_rows - 1) * spacing_h if n_rows > 1 else 0
        total_w_spacing = (n_cols - 1) * spacing_w if n_cols > 1 else 0
        total_height = sum(height_ratios) + total_h_spacing
        total_width = sum(width_ratios) + total_w_spacing

        # Create figure
        fig = plt.figure(figsize=(total_width, total_height))

        # Create GridSpec with proper spacing
        # IMPORTANT: Set left=0, right=1, bottom=0, top=1 to use the FULL figure
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=gs_hspace,
            wspace=gs_wspace,
            left=0,
            right=1,
            bottom=0,
            top=1,
        )

        # Calculate main heatmap position
        main_row = n_top
        main_col = n_left
        ax_main = fig.add_subplot(gs[main_row, main_col])

        # Prepare heatmap params
        heatmap_params = self._heatmap_params.copy()

        # Remove complex-heatmap-specific params that shouldn't go to heatmap()
        heatmap_params.pop('xlabel_side', None)
        heatmap_params.pop('ylabel_side', None)

        # Store and remove title/labels - we'll apply them to the complex heatmap
        title = heatmap_params.pop('title', '')
        xlabel = heatmap_params.pop('xlabel', '')
        ylabel = heatmap_params.pop('ylabel', '')

        if self._row_cluster or self._col_cluster:
            heatmap_params['data'] = self._matrix
            heatmap_params['x'] = None
            heatmap_params['y'] = None
            heatmap_params['value'] = None

        # Draw main heatmap
        heatmap(ax=ax_main, **heatmap_params)

        # Hide heatmap tick labels - they're drawn in dedicated label axes
        # DON'T change the coordinate system - keep heatmap's original ticks at 0.5, 1.5, 2.5...
        ax_main.set_xticklabels([])
        ax_main.set_yticklabels([])

        # Apply title and axis labels to the main heatmap axes
        # This follows PyComplexHeatmap's approach of using standard matplotlib methods
        # TODO: In future, consider dedicated text panels for more control (see annotations.py)
        if title:
            ax_main.set_title(title)
        if xlabel:
            ax_main.set_xlabel(xlabel)
        if ylabel:
            ax_main.set_ylabel(ylabel)

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
                ax.tick_params(labelleft=False)
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
                ax.tick_params(labelright=False)
            axes['right'].append(ax)

        return fig, axes

    def _draw_margin_plot(self, ax: Axes, margin: Dict, position: str):
        """Draw a single margin plot or label axes."""
        func = margin['func']
        data = margin['data']
        kwargs = margin['kwargs'].copy()

        # Special handling for label axes (internal 'labels' marker)
        if func == 'labels':
            self._draw_label_axes(
                ax, data, position,
                kwargs.get('orientation', 'vertical'),
                text_kws=kwargs.get('text_kws')
            )
            return

        # Check if function is dendrogram - auto-provide linkage and orientation
        is_dendrogram = (func is dendrogram_plot or
                        getattr(func, '__name__', '') == 'dendrogram')
        if is_dendrogram:
            # Auto-provide orientation based on position
            if 'orientation' not in kwargs:
                kwargs['orientation'] = position

            # Auto-provide linkage data if not provided
            if 'linkage' not in kwargs and data is None:
                if position in ('top', 'bottom'):
                    kwargs['linkage'] = self._col_linkage
                else:  # left, right
                    kwargs['linkage'] = self._row_linkage

        # Check if function is ticklabels - auto-provide labels and position
        is_ticklabels = (func is ticklabels_plot or
                        getattr(func, '__name__', '') == 'ticklabels')
        if is_ticklabels:
            # Auto-provide position
            if 'position' not in kwargs:
                kwargs['position'] = position

            # Auto-provide orientation based on position
            if 'orientation' not in kwargs:
                kwargs['orientation'] = 'horizontal' if position in ('top', 'bottom') else 'vertical'

            # Auto-provide labels if not provided
            if 'labels' not in kwargs and data is None:
                if position in ('top', 'bottom'):
                    kwargs['labels'] = [str(c) for c in self._matrix.columns]
                else:  # left, right
                    kwargs['labels'] = [str(r) for r in self._matrix.index]

        # Add ax to kwargs
        kwargs['ax'] = ax

        # Add data if provided
        if data is not None:
            kwargs['data'] = data

        # Disable legend by default for margin plots (but not for functions that don't accept it)
        if not is_dendrogram and not is_ticklabels:
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
            for key in ['x', 'y', 'linkage', 'orientation', 'labels', 'position']:
                if key in kwargs:
                    essential[key] = kwargs[key]
            func(**essential)

        # Offset elements to align with heatmap cells
        # Heatmap cells are centered at 0.5, 1.5, 2.5...
        # Categorical plots position elements at 0, 1, 2...
        # Shift by +0.5 to align with cell centers
        if margin['align']:
            orientation = "vertical" if position in ("top", "bottom") else "horizontal"
            # Offset patches (bars, rectangles, etc.)
            if ax.patches:
                offset_patches(ax.patches, offset=0.5, orientation=orientation)
            # Offset lines if any
            if ax.lines:
                offset_lines(ax.lines, offset=0.5, orientation=orientation)
            # Offset collections (scatter plots, etc.)
            if ax.collections:
                offset_collections(ax.collections, offset=0.5, ax=ax, orientation=orientation)

    def _draw_label_axes(self, ax: Axes, labels: List[str], position: str, orientation: str,
                         text_kws: Optional[Dict] = None):
        """Draw tick labels in a dedicated axes.

        Parameters
        ----------
        ax : Axes
            The axes to draw labels in.
        labels : list of str
            The labels to draw.
        position : str
            Position: 'left', 'right', 'top', or 'bottom'.
        orientation : str
            Orientation: 'vertical' or 'horizontal'.
        text_kws : dict, optional
            Keyword arguments passed to ax.text(), including:
            - rotation: angle in degrees
            - va: vertical alignment
            - ha: horizontal alignment
            - fontsize, color, etc.
        """
        # Turn off all axes decorations
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Get default font size based on orientation
        if orientation == 'vertical':
            default_fontsize = resolve_param("ytick.labelsize")
        else:
            default_fontsize = resolve_param("xtick.labelsize")

        # Build final text kwargs with defaults
        final_text_kws = text_kws.copy() if text_kws else {}
        if 'fontsize' not in final_text_kws:
            final_text_kws['fontsize'] = default_fontsize

        n_labels = len(labels)

        if position in ['left', 'right']:
            # Vertical labels for y-axis
            # Match heatmap's coordinate system: cells centered at 0.5, 1.5, 2.5...
            ax.set_ylim(0, n_labels)
            ax.invert_yaxis()  # Match heatmap orientation

            for i, label in enumerate(labels):
                # Draw at cell centers (0.5, 1.5, 2.5...) to match heatmap ticks
                y_pos = i + 0.5
                x_pos = 0 if position == 'right' else 1
                ax.text(x_pos, y_pos, label,
                       transform=ax.get_yaxis_transform(),
                       **final_text_kws)

        else:  # top or bottom
            # Horizontal labels for x-axis
            # Match heatmap's coordinate system: cells centered at 0.5, 1.5, 2.5...
            ax.set_xlim(0, n_labels)

            for i, label in enumerate(labels):
                # Draw at cell centers (0.5, 1.5, 2.5...) to match heatmap ticks
                x_pos = i + 0.5
                y_pos = 1 if position == 'bottom' else 0
                ax.text(x_pos, y_pos, label,
                       transform=ax.get_xaxis_transform(),
                       **final_text_kws)
