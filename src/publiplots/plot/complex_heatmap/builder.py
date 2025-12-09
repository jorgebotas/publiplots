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
from .dendrogram import cluster_data, dendrogram as dendrogram_plot
from .ticklabels import ticklabels as ticklabels_plot

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
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    # Clustering parameters
    row_cluster: bool = False,
    col_cluster: bool = False,
    cluster_method: str = "ward",
    cluster_metric: str = "euclidean",
    # Layout parameters
    hspace: float = 1.0,
    wspace: float = 1.0,
    **kwargs
) -> 'ComplexHeatmapBuilder':
    """
    Create a complex heatmap builder for adding margin plots.

    This function returns a builder object that allows adding plots to the
    margins of a heatmap using method chaining. Tick labels and dendrograms
    can be added via `.add_*()` methods. Call `.build()` to render the final figure.

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
    mask : DataFrame or array, optional
        Cell mask.
    row_cluster : bool, default=False
        Cluster rows. Works with both wide and long format data.
    col_cluster : bool, default=False
        Cluster columns. Works with both wide and long format data.
    cluster_method : str, default="ward"
        Clustering linkage method (ward, single, complete, average).
    cluster_metric : str, default="euclidean"
        Clustering distance metric.
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
    Basic complex heatmap with tick labels and dendrograms:

    >>> fig, axes = (
    ...     pp.complex_heatmap(data, x="sample", y="gene", value="expr",
    ...                        row_cluster=True, col_cluster=True)
    ...     .add_top(pp.dendrogram, height=10, color="black")
    ...     .add_bottom(pp.ticklabels, height=15, rotation=45)
    ...     .add_left(pp.dendrogram, width=10, color="black")
    ...     .add_right(pp.ticklabels, width=20)
    ...     .build()
    ... )

    With additional margin plots:

    >>> fig, axes = (
    ...     pp.complex_heatmap(matrix, row_cluster=True)
    ...     .add_top(pp.barplot, data=totals, x="sample", y="count", height=15)
    ...     .add_left(pp.dendrogram, width=10)
    ...     .add_right(pp.ticklabels, width=20)
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
        mask=mask,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
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
        mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        row_cluster: bool = False,
        col_cluster: bool = False,
        cluster_method: str = "ward",
        cluster_metric: str = "euclidean",
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
        height: Optional[float] = None,
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
            Plot function (e.g., pp.barplot, pp.ticklabels). Must accept `ax` parameter.
        height : float, optional
            Height in millimeters. If None, automatically calculated for ticklabels.
            Default: None for ticklabels, 15 for other plots.
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
        # Auto-detect if this is ticklabels and set default size
        is_ticklabels = (func is ticklabels_plot or
                        getattr(func, '__name__', '') == 'ticklabels')
        if height is None:
            height = 'auto' if is_ticklabels else 15

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
        height: Optional[float] = None,
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
            Plot function (e.g., pp.barplot, pp.ticklabels). Must accept `ax` parameter.
        height : float, optional
            Height in millimeters. If None, automatically calculated for ticklabels.
            Default: None for ticklabels, 15 for other plots.
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
        # Auto-detect if this is ticklabels and set default size
        is_ticklabels = (func is ticklabels_plot or
                        getattr(func, '__name__', '') == 'ticklabels')
        if height is None:
            height = 'auto' if is_ticklabels else 15

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
        width: Optional[float] = None,
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
            Plot function (e.g., pp.barplot, pp.ticklabels). Must accept `ax` parameter.
        width : float, optional
            Width in millimeters. If None, automatically calculated for ticklabels.
            Default: None for ticklabels, 15 for other plots.
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
        # Auto-detect if this is ticklabels and set default size
        is_ticklabels = (func is ticklabels_plot or
                        getattr(func, '__name__', '') == 'ticklabels')
        if width is None:
            width = 'auto' if is_ticklabels else 15

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
        width: Optional[float] = None,
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
            Plot function (e.g., pp.barplot, pp.ticklabels). Must accept `ax` parameter.
        width : float, optional
            Width in millimeters. If None, automatically calculated for ticklabels.
            Default: None for ticklabels, 15 for other plots.
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
        # Auto-detect if this is ticklabels and set default size
        is_ticklabels = (func is ticklabels_plot or
                        getattr(func, '__name__', '') == 'ticklabels')
        if width is None:
            width = 'auto' if is_ticklabels else 15

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

        # Force align=False for dendrograms (they have their own coordinate system)
        # This must happen BEFORE axes are created with sharex/sharey
        for position in ['top', 'bottom', 'left', 'right']:
            for margin in self._margins[position]:
                func = margin['func']
                is_dendrogram = (func is dendrogram_plot or
                                getattr(func, '__name__', '') == 'dendrogram')
                if is_dendrogram:
                    margin['align'] = False

        # Auto-calculate sizes for ticklabels with size='auto'
        # This needs to happen after data preparation so we have the matrix with labels
        from .ticklabels import calculate_ticklabel_space

        # Create a temporary figure for accurate text measurement
        temp_fig = plt.figure(figsize=self._figsize)

        for position in ['top', 'bottom', 'left', 'right']:
            for margin in self._margins[position]:
                if margin['size'] == 'auto':
                    func = margin['func']
                    is_ticklabels = (func is ticklabels_plot or
                                    getattr(func, '__name__', '') == 'ticklabels')
                    if is_ticklabels:
                        # Get labels
                        if position in ('top', 'bottom'):
                            labels = [str(c) for c in matrix.columns]
                        else:  # left, right
                            labels = [str(r) for r in matrix.index]

                        # Extract kwargs for size calculation
                        kwargs = margin['kwargs']
                        rotation = kwargs.get('rotation', 0)
                        fontsize = kwargs.get('fontsize', None)

                        # Calculate required space
                        space_mm = calculate_ticklabel_space(
                            labels=labels,
                            position=position,
                            rotation=rotation,
                            fontsize=fontsize,
                            fig=temp_fig,
                            **{k: v for k, v in kwargs.items() if k not in ['rotation', 'fontsize', 'ax', 'labels', 'position', 'orientation']}
                        )

                        # Update margin size
                        margin['size'] = space_mm

        # Close temp figure
        plt.close(temp_fig)

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
        """Draw a single margin plot."""
        func = margin['func']
        data = margin['data']
        kwargs = margin['kwargs'].copy()

        # Check if function is dendrogram - auto-provide linkage and orientation
        is_dendrogram = (func is dendrogram_plot or
                        getattr(func, '__name__', '') == 'dendrogram')
        if is_dendrogram:
            # Auto-provide orientation based on position
            if 'orientation' not in kwargs:
                kwargs['orientation'] = position

            # Auto-provide linkage or data if not provided
            if 'linkage' not in kwargs and data is None:
                if position in ('top', 'bottom'):
                    linkage = self._col_linkage
                    cluster_param = 'col_cluster'
                else:  # left, right
                    linkage = self._row_linkage
                    cluster_param = 'row_cluster'

                # Check if clustering was enabled
                if linkage is None:
                    raise ValueError(
                        f"Cannot add dendrogram to {position} margin: clustering is disabled. "
                        f"Either enable clustering by setting {cluster_param}=True in complex_heatmap(), "
                        f"or provide custom linkage data via the 'linkage' parameter."
                    )

                kwargs['linkage'] = linkage

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

