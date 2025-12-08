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
from .dendrogram import cluster_data, dendrogram as dendrogram_plot

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
    xlabel_side: str = "auto",
    ylabel_side: str = "auto",
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
    xlabel_side : str, default="auto"
        X-axis label position: 'top', 'bottom', or 'auto'.
    ylabel_side : str, default="auto"
        Y-axis label position: 'left', 'right', or 'auto'.
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
        xlabel_side=xlabel_side,
        ylabel_side=ylabel_side,
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
        xlabel_side: str = "auto",
        ylabel_side: str = "auto",
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
            "xlabel_side": xlabel_side,
            "ylabel_side": ylabel_side,
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

        # Auto-detect label sides
        xlabel_side = self._heatmap_params['xlabel_side']
        ylabel_side = self._heatmap_params['ylabel_side']

        if ylabel_side == 'auto':
            # If there are left margins, put ylabel on right
            ylabel_side = 'right' if self._margins['left'] else 'left'

        if xlabel_side == 'auto':
            # If there are top margins, put xlabel on bottom
            xlabel_side = 'bottom' if self._margins['top'] else 'top'

        # Label space allocation (mm)
        LABEL_SPACE = 12  # Space for tick labels in mm

        # Add space for labels when they're on the opposite side
        # This prevents overlap with margin plots
        label_space_right = LABEL_SPACE * MM2INCH if ylabel_side == 'right' else 0
        label_space_bottom = LABEL_SPACE * MM2INCH if xlabel_side == 'bottom' else 0
        label_space_left = LABEL_SPACE * MM2INCH if ylabel_side == 'left' else 0
        label_space_top = LABEL_SPACE * MM2INCH if xlabel_side == 'top' else 0

        # Count margin plots
        n_top = len(self._margins['top'])
        n_bottom = len(self._margins['bottom'])
        n_left = len(self._margins['left'])
        n_right = len(self._margins['right'])

        # Calculate grid dimensions
        # Add extra rows/cols for label space
        has_label_col_right = (ylabel_side == 'right' and n_right > 0)
        has_label_col_left = (ylabel_side == 'left' and n_left > 0)
        has_label_row_bottom = (xlabel_side == 'bottom' and n_bottom > 0)
        has_label_row_top = (xlabel_side == 'top' and n_top > 0)

        n_rows = n_top + 1 + n_bottom + (1 if has_label_row_bottom else 0) + (1 if has_label_row_top else 0)
        n_cols = n_left + 1 + n_right + (1 if has_label_col_right else 0) + (1 if has_label_col_left else 0)

        # Calculate size ratios
        # Convert mm sizes to relative ratios
        main_height = self._figsize[1]  # inches
        main_width = self._figsize[0]   # inches

        # Top margins (in reverse order - first added is closest to heatmap)
        top_sizes = [m['size'] * MM2INCH for m in reversed(self._margins['top'])]
        bottom_sizes = [m['size'] * MM2INCH for m in self._margins['bottom']]
        left_sizes = [m['size'] * MM2INCH for m in reversed(self._margins['left'])]
        right_sizes = [m['size'] * MM2INCH for m in self._margins['right']]

        # Build height and width ratios with label spaces
        height_ratios = []
        if has_label_row_top:
            height_ratios.append(label_space_top)
        height_ratios.extend(top_sizes)
        height_ratios.append(main_height)
        height_ratios.extend(bottom_sizes)
        if has_label_row_bottom:
            height_ratios.append(label_space_bottom)

        width_ratios = []
        if has_label_col_left:
            width_ratios.append(label_space_left)
        width_ratios.extend(left_sizes)
        width_ratios.append(main_width)
        width_ratios.extend(right_sizes)
        if has_label_col_right:
            width_ratios.append(label_space_right)

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
            hspace=self._hspace * MM2INCH / main_height if main_height > 0 else 0,
            wspace=self._wspace * MM2INCH / main_width if main_width > 0 else 0,
        )

        # Calculate main heatmap position accounting for label spaces
        main_row = (1 if has_label_row_top else 0) + n_top
        main_col = (1 if has_label_col_left else 0) + n_left
        ax_main = fig.add_subplot(gs[main_row, main_col])

        # Prepare heatmap params for the clustered data
        heatmap_params = self._heatmap_params.copy()

        # Remove label side parameters (these are builder-specific, not heatmap params)
        heatmap_params.pop('xlabel_side', None)
        heatmap_params.pop('ylabel_side', None)

        if self._row_cluster or self._col_cluster:
            # Use the clustered matrix directly (wide format)
            heatmap_params['data'] = self._matrix
            heatmap_params['x'] = None
            heatmap_params['y'] = None
            heatmap_params['value'] = None

        # Draw main heatmap
        heatmap(ax=ax_main, **heatmap_params)

        # Set ylabel and xlabel sides
        if ylabel_side == 'right':
            ax_main.yaxis.tick_right()
            ax_main.yaxis.set_label_position('right')

        if xlabel_side == 'top':
            ax_main.xaxis.tick_top()
            ax_main.xaxis.set_label_position('top')

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
            row_idx = (1 if has_label_row_top else 0) + n_top - 1 - i
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
            row_idx = (1 if has_label_row_top else 0) + n_top + 1 + i
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
            col_idx = (1 if has_label_col_left else 0) + n_left - 1 - i
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
            # Right margins start after the main column
            # If there's a label column on right, margins go after it
            col_idx = (1 if has_label_col_left else 0) + n_left + 1 + i
            ax = fig.add_subplot(
                gs[main_row, col_idx],
                sharey=ax_main if margin['align'] else None
            )
            self._draw_margin_plot(ax, margin, 'right')
            if margin['align'] and i < len(self._margins['right']) - 1:
                ax.tick_params(labelright=False)
            axes['right'].append(ax)

        # Adjust tick label visibility based on label position
        if ylabel_side == 'right':
            # Hide left tick labels
            ax_main.tick_params(labelleft=False)
        elif any(m['align'] for m in self._margins['left']):
            # Hide left tick labels if there are aligned left margins
            ax_main.tick_params(labelleft=False)

        if xlabel_side == 'top':
            # Hide bottom tick labels
            ax_main.tick_params(labelbottom=False)
        elif any(m['align'] for m in self._margins['top']):
            # Hide top tick labels if there are aligned top margins
            ax_main.tick_params(labeltop=False)

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

        # Fix coordinate alignment for categorical plots
        # Heatmaps use cell-centered coordinates (0.5, 1.5, 2.5, ...)
        # But categorical plots use integer positions (0, 1, 2, ...)
        # We need to shift categorical plot positions by +0.5 to align
        if margin['align']:
            if position in ['top', 'bottom']:
                # Horizontal alignment - adjust x-axis
                # Check if x-axis has integer-like positions (categorical data)
                xticks = ax.get_xticks()
                # Check if ticks are close to integers (within 0.01)
                if len(xticks) > 1 and np.allclose(xticks, np.round(xticks), atol=0.01):
                    # Also check that they're not already cell-centered (not x.5)
                    if not np.allclose(xticks - np.floor(xticks), 0.5, atol=0.01):
                        # Shift all bars/patches by +0.5
                        for patch in ax.patches:
                            if hasattr(patch, 'get_x'):
                                patch.set_x(patch.get_x() + 0.5)
                        # Shift scatter points
                        for collection in ax.collections:
                            if hasattr(collection, 'get_offsets'):
                                offsets = collection.get_offsets()
                                if len(offsets) > 0:
                                    offsets[:, 0] += 0.5
                                    collection.set_offsets(offsets)

            elif position in ['left', 'right']:
                # Vertical alignment - adjust y-axis
                yticks = ax.get_yticks()
                # Check if ticks are close to integers (categorical data)
                if len(yticks) > 1 and np.allclose(yticks, np.round(yticks), atol=0.01):
                    # Also check that they're not already cell-centered (not x.5)
                    if not np.allclose(yticks - np.floor(yticks), 0.5, atol=0.01):
                        # Shift all bars/patches by +0.5
                        for patch in ax.patches:
                            if hasattr(patch, 'get_y'):
                                patch.set_y(patch.get_y() + 0.5)
                        # Shift scatter points
                        for collection in ax.collections:
                            if hasattr(collection, 'get_offsets'):
                                offsets = collection.get_offsets()
                                if len(offsets) > 0:
                                    offsets[:, 1] += 0.5
                                    collection.set_offsets(offsets)
