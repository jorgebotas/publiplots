"""
Annotation functions for complex heatmaps.

This module provides annotation types that can be added to margins of complex
heatmaps, inspired by PyComplexHeatmap. All annotations use the publiplots
double-layer rendering style (transparent fill + opaque edge).
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable, get_cmap
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple

from publiplots.themes.rcparams import resolve_param
from publiplots.themes.colors import resolve_palette_map
from publiplots.utils import is_categorical
from publiplots.utils.transparency import apply_transparency
from publiplots.utils.legend import legend as legend_builder


def block(
    data: Union[pd.Series, pd.DataFrame, List, np.ndarray],
    # Orientation and position
    orientation: str = "horizontal",
    # Color encoding
    cmap: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colors: Optional[List[str]] = None,
    # Styling
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    # Legend
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    # Text overlay
    text: bool = False,
    text_kws: Optional[Dict] = None,
    # Axes
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create simple annotation blocks (colored rectangles) for categorical or
    continuous variables.

    This creates a single row or column of colored blocks, similar to
    PyComplexHeatmap's anno_simple. Perfect for showing metadata like sample
    groups, tissue types, or any categorical/continuous variable.

    Uses publiplots' double-layer rendering: transparent fill + opaque edge.

    Parameters
    ----------
    data : Series, DataFrame, list, or array
        Data to visualize as blocks. Can be:
        - Series: Single row/column of categorical or continuous values
        - DataFrame: Multiple rows/columns stacked
        - List/array: Single row/column
    orientation : {'horizontal', 'vertical'}, default='horizontal'
        Block orientation:
        - 'horizontal': Blocks arranged left-to-right (for top/bottom margins)
        - 'vertical': Blocks arranged top-to-bottom (for left/right margins)
    cmap : str, optional
        Colormap for continuous data. Default: 'viridis' for continuous.
    palette : str, dict, or list, optional
        Color palette for categorical data:
        - str: Palette name (e.g., 'Set2', 'pastel')
        - dict: Mapping from categories to colors
        - list: List of colors
    vmin, vmax : float, optional
        Value range for continuous data colormapping.
    colors : list of str, optional
        Explicit colors for each block (overrides cmap/palette).
    alpha : float, optional
        Face transparency (0-1). Uses rcParams default. Edge always opaque.
    linewidth : float, optional
        Edge linewidth. Uses rcParams default.
    edgecolor : str, optional
        Edge color. If None, uses face color.
    legend : bool, default=True
        Whether to show legend (stored for later legend building).
    legend_kws : dict, optional
        Legend customization (label, title, etc.).
    text : bool, default=False
        Whether to overlay text labels on blocks.
    text_kws : dict, optional
        Text styling (fontsize, color, ha, va, etc.).
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments (for future extensions).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.

    Examples
    --------
    Categorical blocks (horizontal):

    >>> tissue_types = pd.Series(['Brain', 'Liver', 'Heart', 'Brain', 'Liver'])
    >>> fig, ax = pp.block(tissue_types, palette='Set2')

    Continuous blocks (vertical):

    >>> quality_scores = pd.Series([0.95, 0.87, 0.92, 0.78])
    >>> fig, ax = pp.block(quality_scores, orientation='vertical',
    ...                     cmap='RdYlGn', vmin=0, vmax=1)

    Multiple rows stacked:

    >>> metadata = pd.DataFrame({
    ...     'tissue': ['Brain', 'Liver', 'Heart'],
    ...     'condition': ['Control', 'Treatment', 'Control']
    ... })
    >>> fig, ax = pp.block(metadata, palette='pastel')

    With text overlay:

    >>> groups = pd.Series(['A', 'B', 'C', 'A', 'B'])
    >>> fig, ax = pp.block(groups, text=True, text_kws={'fontsize': 10})

    Notes
    -----
    - Designed for use with complex_heatmap margins, but works standalone
    - Automatically detects categorical vs continuous data
    - Uses publiplots transparency styling (transparent fill + opaque edge)
    - Legend metadata stored on axes for complex_heatmap legend reconciliation
    """
    # Resolve defaults from rcParams
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)

    # Create figure if needed
    if ax is None:
        figsize = (6, 0.5) if orientation == 'horizontal' else (0.5, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Convert data to DataFrame for uniform handling
    if isinstance(data, pd.Series):
        data_df = data.to_frame().T if orientation == 'horizontal' else data.to_frame()
    elif isinstance(data, pd.DataFrame):
        data_df = data if orientation == 'horizontal' else data.T
    elif isinstance(data, (list, np.ndarray)):
        data_array = np.array(data)
        if data_array.ndim == 1:
            data_df = pd.DataFrame([data_array] if orientation == 'horizontal' else [[d] for d in data_array])
        else:
            data_df = pd.DataFrame(data_array)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    n_rows, n_cols = data_df.shape

    # Determine if data is categorical or continuous
    # Check the flattened values, not the DataFrame structure
    flat_values = data_df.values.ravel()
    categorical = is_categorical(flat_values)

    # Prepare colors and normalization
    color_map = None
    norm = None
    cmap_obj = None

    if colors is not None:
        # Use explicit colors
        color_values = colors
    elif categorical:
        # Categorical coloring
        unique_values = pd.unique(data_df.values.ravel())

        if palette is None:
            palette = 'pastel'

        color_map = resolve_palette_map(
            values=unique_values,
            palette=palette
        )
        color_values = [[color_map[val] for val in row] for row in data_df.values]
    else:
        # Continuous coloring
        if cmap is None:
            cmap = 'viridis'

        # Flatten values and get min/max
        flat_values = data_df.values.ravel()
        v_min = vmin if vmin is not None else np.nanmin(flat_values)
        v_max = vmax if vmax is not None else np.nanmax(flat_values)

        # Create normalizer and colormap
        norm = Normalize(vmin=v_min, vmax=v_max)
        cmap_obj = get_cmap(cmap)

        # Map values to colors
        color_values = [[cmap_obj(norm(val)) for val in row] for row in data_df.values]

    # Draw rectangles
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            if orientation == 'horizontal':
                x, y, w, h = j, n_rows - i - 1, 1, 1
            else:
                x, y, w, h = i, j, 1, 1

            # Get color for this block
            if isinstance(color_values[0], list):
                color = color_values[i][j]
            else:
                color = color_values[j] if orientation == 'horizontal' else color_values[i]

            # Create rectangle with double-layer rendering
            # Layer 1: Transparent fill
            rect_fill = Rectangle(
                (x, y), w, h,
                facecolor=color,
                edgecolor='none',
                linewidth=0,
                zorder=1
            )
            ax.add_patch(rect_fill)

            # Layer 2: Opaque edge
            edge_c = edgecolor if edgecolor else color
            rect_edge = Rectangle(
                (x, y), w, h,
                facecolor='none',
                edgecolor=edge_c,
                linewidth=linewidth,
                zorder=2
            )
            ax.add_patch(rect_edge)

            patches.append(rect_fill)

            # Add text overlay if requested
            if text:
                val = data_df.iloc[i, j]
                text_kwargs = text_kws or {}
                text_defaults = {
                    'fontsize': resolve_param("font.size"),
                    'ha': 'center',
                    'va': 'center',
                    'color': 'black',
                    'weight': 'normal'
                }
                text_defaults.update(text_kwargs)

                ax.text(
                    x + w/2, y + h/2, str(val),
                    **text_defaults,
                    zorder=3
                )

    # Apply transparency using publiplots utility
    apply_transparency(patches, face_alpha=alpha, edge_alpha=1.0)

    # Set axis limits and appearance
    if orientation == 'horizontal':
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()
    else:
        ax.set_xlim(0, n_rows)
        ax.set_ylim(0, n_cols)
        ax.invert_yaxis()

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Store legend metadata on axes for complex_heatmap reconciliation
    if legend:
        legend_kws = legend_kws or {}
        label = legend_kws.get('label', '')

        if categorical:
            # Discrete legend
            from publiplots.utils import create_legend_handles
            unique_values = pd.unique(data_df.values.ravel())
            handles = create_legend_handles(
                labels=[str(v) for v in unique_values],
                colors=[color_map[v] for v in unique_values],
                alpha=alpha,
                linewidth=linewidth,
                style='rectangle'
            )

            # Store for later
            if not hasattr(ax.patches[0], '_legend_data'):
                ax.patches[0]._legend_data = {}
            ax.patches[0]._legend_data['hue'] = {
                'handles': handles,
                'label': label
            }
        else:
            # Continuous colorbar
            mappable = ScalarMappable(norm=norm, cmap=cmap_obj)

            # Store for later
            if not hasattr(ax.patches[0], '_legend_data'):
                ax.patches[0]._legend_data = {}
            ax.patches[0]._legend_data['hue'] = {
                'type': 'colorbar',
                'mappable': mappable,
                'label': label
            }

    return fig, ax


def label(
    labels: Union[List[str], pd.Series, np.ndarray],
    # Orientation and position
    orientation: str = "horizontal",
    # Text styling
    rotation: float = 0,
    fontsize: Optional[float] = None,
    ha: str = "center",
    va: str = "center",
    color: Optional[str] = None,
    fontweight: str = "normal",
    # Arrow connection
    arrow: bool = False,
    arrow_kws: Optional[Dict] = None,
    # Positioning
    offset: float = 0.5,
    # Axes
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create text label annotations with optional arrow connections.

    Similar to PyComplexHeatmap's anno_label, this function creates text labels
    that can be positioned along heatmap margins, with optional arrows pointing
    to specific positions. Perfect for gene names, sample IDs, or any textual
    annotations.

    Parameters
    ----------
    labels : list, Series, or array
        Text labels to display.
    orientation : {'horizontal', 'vertical'}, default='horizontal'
        Label orientation:
        - 'horizontal': Labels arranged left-to-right (for top/bottom margins)
        - 'vertical': Labels arranged top-to-bottom (for left/right margins)
    rotation : float, default=0
        Text rotation angle in degrees.
        Common values: 0, 45, 90, -45, -90.
    fontsize : float, optional
        Font size. Uses rcParams default if None.
    ha : str, default='center'
        Horizontal alignment: 'left', 'center', 'right'.
    va : str, default='center'
        Vertical alignment: 'top', 'center', 'bottom', 'baseline'.
    color : str, optional
        Text color. Uses rcParams default if None.
    fontweight : str, default='normal'
        Font weight: 'normal', 'bold', 'light', 'heavy'.
    arrow : bool, default=False
        Whether to draw connecting arrows from labels to positions.
    arrow_kws : dict, optional
        Arrow styling:
        - arrowstyle: Arrow style (e.g., '->', '-|>', 'fancy')
        - color: Arrow color
        - linewidth: Arrow line width
        - shrinkA, shrinkB: Padding at arrow ends
    offset : float, default=0.5
        Distance of labels from axis (in data coordinates).
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to ax.text().

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.

    Examples
    --------
    Simple horizontal labels:

    >>> gene_names = ['BRCA1', 'TP53', 'EGFR', 'MYC']
    >>> fig, ax = pp.label(gene_names, rotation=45, ha='right')

    Vertical labels with arrows:

    >>> sample_ids = ['S001', 'S002', 'S003']
    >>> fig, ax = pp.label(sample_ids, orientation='vertical',
    ...                     arrow=True, arrow_kws={'arrowstyle': '->'})

    Custom styling:

    >>> categories = ['Group A', 'Group B', 'Group C']
    >>> fig, ax = pp.label(categories, fontsize=12, fontweight='bold',
    ...                     color='darkblue', rotation=90)

    Notes
    -----
    - Designed for complex_heatmap margins but works standalone
    - Arrow connections useful for sparse labeling (not every position)
    - Rotation is applied after text is positioned
    - For dense labels, consider using pp.ticklabels instead
    """
    # Resolve defaults
    fontsize = resolve_param("font.size", fontsize)
    color = resolve_param("color", color)

    # Convert labels to list
    if isinstance(labels, pd.Series):
        labels = labels.tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = list(labels)

    n_labels = len(labels)

    # Create figure if needed
    if ax is None:
        figsize = (6, 0.8) if orientation == 'horizontal' else (0.8, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Position labels
    for i, lbl in enumerate(labels):
        if orientation == 'horizontal':
            # Labels below axis (for top/bottom margins)
            x = i + 0.5
            y = offset

            # Adjust alignment for rotation
            if rotation > 0:  # Rotated right
                text_ha = ha if ha != 'center' else 'right'
                text_va = va if va != 'center' else 'bottom'
            elif rotation < 0:  # Rotated left
                text_ha = ha if ha != 'center' else 'left'
                text_va = va if va != 'center' else 'bottom'
            else:  # No rotation
                text_ha = ha
                text_va = va

        else:  # vertical
            # Labels to right of axis (for left/right margins)
            x = offset
            y = i + 0.5

            # Adjust alignment for vertical orientation
            if rotation == 0:
                text_ha = ha if ha != 'center' else 'left'
                text_va = va if va != 'center' else 'center'
            else:
                text_ha = ha
                text_va = va

        # Draw text
        text_kwargs = kwargs.copy()
        text_kwargs.update({
            'fontsize': fontsize,
            'ha': text_ha,
            'va': text_va,
            'color': color,
            'weight': fontweight,
            'rotation': rotation,
        })

        ax.text(x, y, str(lbl), **text_kwargs, zorder=2)

        # Draw arrow if requested
        if arrow:
            arrow_kwargs = arrow_kws or {}
            arrow_defaults = {
                'arrowstyle': '->',
                'color': color or 'black',
                'linewidth': 1,
                'shrinkA': 2,
                'shrinkB': 2,
            }
            arrow_defaults.update(arrow_kwargs)

            # Determine arrow endpoints
            if orientation == 'horizontal':
                # Arrow from text to axis
                arrow_start = (x, y - 0.1)
                arrow_end = (x, 0)
            else:
                # Arrow from text to axis
                arrow_start = (x - 0.1, y)
                arrow_end = (0, y)

            arrow_patch = FancyArrowPatch(
                arrow_start, arrow_end,
                **arrow_defaults,
                zorder=1
            )
            ax.add_patch(arrow_patch)

    # Set axis limits
    if orientation == 'horizontal':
        ax.set_xlim(0, n_labels)
        ax.set_ylim(0, offset + 0.5)
        ax.invert_yaxis()
    else:
        ax.set_xlim(0, offset + 0.5)
        ax.set_ylim(0, n_labels)
        ax.invert_yaxis()

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def spacer(
    height: Optional[float] = None,
    width: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create an empty spacer annotation for visual separation.

    Similar to PyComplexHeatmap's anno_spacer, this creates blank space between
    annotations in complex heatmap margins. Useful for grouping related
    annotations or improving visual clarity.

    Parameters
    ----------
    height : float, optional
        Height in millimeters (for horizontal spacers in top/bottom margins).
        Only used when creating new figure. Ignored when ax is provided.
    width : float, optional
        Width in millimeters (for vertical spacers in left/right margins).
        Only used when creating new figure. Ignored when ax is provided.
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure with specified dimensions.
    **kwargs
        Additional keyword arguments (for future extensions).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.

    Examples
    --------
    Horizontal spacer (for top/bottom margins):

    >>> fig, axes = (
    ...     pp.complex_heatmap(data, x='sample', y='gene', value='expr')
    ...     .add_top(pp.block, data=tissue, height=5)
    ...     .add_top(pp.spacer, height=3)  # Add 3mm gap
    ...     .add_top(pp.block, data=treatment, height=5)
    ...     .build()
    ... )

    Vertical spacer (for left/right margins):

    >>> fig, axes = (
    ...     pp.complex_heatmap(data, x='sample', y='gene', value='expr')
    ...     .add_right(pp.label, labels=gene_names, width=20)
    ...     .add_right(pp.spacer, width=5)  # Add 5mm gap
    ...     .add_right(pp.barplot, data=scores, width=15)
    ...     .build()
    ... )

    Notes
    -----
    - Designed for complex_heatmap margins
    - Size is controlled by height/width parameters in .add_<margin>() calls
    - Creates completely empty axes (no spines, ticks, or content)
    - Useful for improving readability of complex multi-annotation layouts
    """
    # Create figure if needed
    if ax is None:
        # Convert mm to inches
        MM2INCH = 1 / 25.4
        h_inch = (height * MM2INCH) if height else 0.3
        w_inch = (width * MM2INCH) if width else 0.3
        fig, ax = plt.subplots(figsize=(w_inch, h_inch))
    else:
        fig = ax.get_figure()

    # Make completely blank
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove any padding
    ax.set_aspect('auto')

    return fig, ax


__all__ = [
    "block",
    "label",
    "spacer",
]
