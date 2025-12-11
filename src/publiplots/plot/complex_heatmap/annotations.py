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


def _prepare_annotation_data(
    data: Union[pd.Series, pd.DataFrame, List, np.ndarray],
    x: Optional[Union[str, bool]] = None,
    y: Optional[Union[str, bool]] = None,
    order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, bool, int, int]:
    """
    Prepare data for annotation functions (block, label, etc.).

    This helper function handles all common data preprocessing:
    - Determines orientation (horizontal vs vertical)
    - Extracts data from various input formats (DataFrame, Series, array)
    - Handles metadata column mapping (e.g., x="condition" when order has sample names)
    - Returns properly formatted DataFrame

    Parameters
    ----------
    data : Series, DataFrame, list, or array
        Input data to process.
    x : str or bool, optional
        Column name (if DataFrame) or True for horizontal orientation.
    y : str or bool, optional
        Column name (if DataFrame) or True for vertical orientation.
    order : list of str, optional
        Order of positions (typically from heatmap columns/index).

    Returns
    -------
    data_df : DataFrame
        Processed data as DataFrame (1 row for horizontal, n rows for vertical).
    horizontal : bool
        True if horizontal orientation, False if vertical.
    n_rows : int
        Number of rows in data_df.
    n_cols : int
        Number of columns in data_df.

    Examples
    --------
    >>> # Extract from DataFrame column
    >>> df = pd.DataFrame({'sample': ['S1', 'S2'], 'condition': ['A', 'B']})
    >>> data_df, horiz, nr, nc = _prepare_annotation_data(df, x='condition', order=['S1', 'S2'])

    >>> # Wide format matrix
    >>> matrix = pd.DataFrame([[1, 2]], columns=['S1', 'S2'])
    >>> data_df, horiz, nr, nc = _prepare_annotation_data(matrix, x=True, order=['S1', 'S2'])
    """
    # Determine orientation from x/y parameters
    if x is not None and y is not None:
        raise ValueError("Only one of x or y should be set, not both")

    # Infer orientation: x → horizontal, y → vertical
    if x is not None:
        horizontal = True
    elif y is not None:
        horizontal = False
    elif order is not None and isinstance(data, pd.DataFrame):
        # Infer from order matching columns (horizontal) or index (vertical)
        horizontal = list(data.columns) == list(order) or len(data.columns) == len(order)
    else:
        # Default to horizontal for Series/array
        horizontal = True

    # Extract and prepare data
    if isinstance(data, pd.DataFrame):
        # Case 1: DataFrame with column name to extract (long format)
        if isinstance(x, str):
            # Note: merge parameter affects drawing, not data extraction
            # Always extract one value per position in order
            if order is not None:
                # Check if x column values match order (axis column like 'sample')
                x_values_match_order = any(val in data[x].values for val in order)

                if x_values_match_order:
                    # x is the axis column - extract in order
                    data_filtered = data[data[x].isin(order)]
                    data_series = data_filtered[x]
                    # Get unique values in order
                    seen = set()
                    unique_ordered = []
                    for val in order:
                        if val not in seen and val in data_series.values:
                            unique_ordered.append(val)
                            seen.add(val)
                    data_series = pd.Series(unique_ordered)
                else:
                    # x is a metadata column - need to map from axis to metadata
                    # Find axis column (column whose unique values match order)
                    axis_col = None
                    for col in data.columns:
                        if col != x:
                            col_unique = set(data[col].unique())
                            if set(order).issubset(col_unique) or col_unique.issubset(set(order)):
                                axis_col = col
                                break

                    if axis_col is not None:
                        # Map from order to x values via axis column
                        mapped_values = []
                        for axis_val in order:
                            # Get rows for this axis value
                            rows = data[data[axis_col] == axis_val]
                            if len(rows) > 0:
                                # Get the x value (assume same for all rows with this axis value)
                                mapped_values.append(rows[x].iloc[0])
                        data_series = pd.Series(mapped_values)
                    else:
                        # Fallback: couldn't find axis column
                        data_series = data[x].unique()
                        data_series = pd.Series(data_series)
            else:
                data_series = data[x].unique()
                data_series = pd.Series(data_series)

            data_df = data_series.to_frame().T

        elif isinstance(y, str):
            # Note: merge parameter affects drawing, not data extraction
            # Always extract one value per position in order
            if order is not None:
                # Check if y column values match order (axis column like 'gene')
                y_values_match_order = any(val in data[y].values for val in order)

                if y_values_match_order:
                    # y is the axis column - extract in order
                    data_filtered = data[data[y].isin(order)]
                    data_series = data_filtered[y]
                    # Get unique values in order
                    seen = set()
                    unique_ordered = []
                    for val in order:
                        if val not in seen and val in data_series.values:
                            unique_ordered.append(val)
                            seen.add(val)
                    data_series = pd.Series(unique_ordered)
                else:
                    # y is a metadata column - need to map from axis to metadata
                    # Find axis column (column whose unique values match order)
                    axis_col = None
                    for col in data.columns:
                        if col != y:
                            col_unique = set(data[col].unique())
                            if set(order).issubset(col_unique) or col_unique.issubset(set(order)):
                                axis_col = col
                                break

                    if axis_col is not None:
                        # Map from order to y values via axis column
                        mapped_values = []
                        for axis_val in order:
                            # Get rows for this axis value
                            rows = data[data[axis_col] == axis_val]
                            if len(rows) > 0:
                                # Get the y value (assume same for all rows with this axis value)
                                mapped_values.append(rows[y].iloc[0])
                        data_series = pd.Series(mapped_values)
                    else:
                        # Fallback: couldn't find axis column
                        data_series = data[y].unique()
                        data_series = pd.Series(data_series)
            else:
                data_series = data[y].unique()
                data_series = pd.Series(data_series)

            data_df = data_series.to_frame()

        else:
            # Case 2: DataFrame as matrix (wide format from complex_heatmap)
            if horizontal:
                # Use columns for horizontal (top/bottom)
                if order is not None:
                    # Filter and reorder columns based on order
                    data_series = pd.Series([col for col in order if col in data.columns])
                    # Fallback: if filtering produced empty result, use all columns
                    if len(data_series) == 0:
                        data_series = pd.Series(data.columns)
                else:
                    data_series = pd.Series(data.columns)
                data_df = data_series.to_frame().T
            else:
                # Use index for vertical (left/right)
                if order is not None:
                    # Filter and reorder index based on order
                    data_series = pd.Series([idx for idx in order if idx in data.index])
                    # Fallback: if filtering produced empty result, use all index
                    if len(data_series) == 0:
                        data_series = pd.Series(data.index)
                else:
                    data_series = pd.Series(data.index)
                data_df = data_series.to_frame()

    elif isinstance(data, pd.Series):
        # Series data
        if order is not None:
            data = data.reindex(order)
        data_df = data.to_frame().T if horizontal else data.to_frame()

    elif isinstance(data, (list, np.ndarray)):
        # Array/list data
        data_array = np.array(data)
        if data_array.ndim == 1:
            if order is not None and len(order) == len(data_array):
                data_array = data_array[np.argsort(order)]
            data_df = pd.DataFrame([data_array] if horizontal else [[d] for d in data_array])
        else:
            data_df = pd.DataFrame(data_array)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    n_rows, n_cols = data_df.shape

    return data_df, horizontal, n_rows, n_cols


def block(
    data: Union[pd.Series, pd.DataFrame, List, np.ndarray],
    # Position/orientation (optional when called from complex_heatmap)
    x: Optional[Union[str, bool]] = None,
    y: Optional[Union[str, bool]] = None,
    # Data handling
    order: Optional[List[str]] = None,
    merge: bool = True,
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
        - DataFrame: Multiple rows/columns stacked (with column name in x/y)
        - List/array: Single row/column
    x : str or bool, optional
        Column name (if data is DataFrame) or True for horizontal orientation.
        Use for top/bottom margins. Can be omitted when called from complex_heatmap
        builder (orientation inferred from position).
    y : str or bool, optional
        Column name (if data is DataFrame) or True for vertical orientation.
        Use for left/right margins. Can be omitted when called from complex_heatmap.
    order : list of str, optional
        Order of categories (passed automatically by complex_heatmap builder).
    merge : bool, default=True
        If True and x/y is a column name, merge values from that column
        (useful for long-format data with multiple rows per category).
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
    Categorical blocks (horizontal, for top/bottom margins):

    >>> tissue_types = pd.Series(['Brain', 'Liver', 'Heart', 'Brain', 'Liver'])
    >>> fig, ax = pp.block(tissue_types, x=True, palette='Set2')

    Continuous blocks (vertical, for left/right margins):

    >>> quality_scores = pd.Series([0.95, 0.87, 0.92, 0.78])
    >>> fig, ax = pp.block(quality_scores, y=True,
    ...                     cmap='RdYlGn', vmin=0, vmax=1)

    Using DataFrame with column names:

    >>> metadata = pd.DataFrame({
    ...     'sample': ['S1', 'S2', 'S3'],
    ...     'tissue': ['Brain', 'Liver', 'Heart'],
    ...     'condition': ['Control', 'Treatment', 'Control']
    ... })
    >>> fig, ax = pp.block(metadata, x='sample', y='tissue', palette='pastel')

    With text overlay:

    >>> groups = pd.Series(['A', 'B', 'C', 'A', 'B'])
    >>> fig, ax = pp.block(groups, x=True, text=True, text_kws={'fontsize': 10})

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

    # Prepare data using helper function
    data_df, horizontal, n_rows, n_cols = _prepare_annotation_data(data, x, y, order)

    # Create figure if needed
    if ax is None:
        figsize = (6, 0.5) if horizontal else (0.5, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Debug: Check if data_df is empty
    if n_rows == 0 or n_cols == 0:
        import warnings
        warnings.warn(
            f"block() received empty data: n_rows={n_rows}, n_cols={n_cols}. "
            f"data_df shape: {data_df.shape}, values: {data_df.values if n_rows > 0 and n_cols > 0 else 'empty'}. "
            f"x={x}, y={y}, horizontal={horizontal}, order={'provided' if order is not None else 'None'}",
            UserWarning
        )

    # Set axis limits early so we can convert linewidth from points to data coordinates
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    if horizontal:
        ax.invert_yaxis()

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

    # Convert linewidth from points to data coordinates
    # This ensures linewidth grows inward, keeping all edges visible
    if linewidth > 0:
        # Get figure and axes info
        fig = ax.get_figure()
        bbox = ax.get_position()  # Axes position in figure coordinates (0-1)
        fig_width_inches, fig_height_inches = fig.get_size_inches()

        # Calculate axes size in inches
        ax_width_inches = bbox.width * fig_width_inches
        ax_height_inches = bbox.height * fig_height_inches

        # Data range (already set above)
        data_width = n_cols  # xlim is (0, n_cols)
        data_height = n_rows  # ylim is (0, n_rows)

        # Convert linewidth from points to data coordinates
        # 72 points per inch
        points_per_inch = 72
        lw_data_x = linewidth * data_width / (ax_width_inches * points_per_inch)
        lw_data_y = linewidth * data_height / (ax_height_inches * points_per_inch)

        # Inset by half linewidth so edge grows inward
        inset_x = lw_data_x / 2
        inset_y = lw_data_y / 2
    else:
        inset_x = 0
        inset_y = 0

    # Build list of rectangles to draw (with merging if requested)
    rectangles_to_draw = []

    if merge:
        # Merge consecutive identical values into single rectangles
        for i in range(n_rows):
            j = 0
            while j < n_cols:
                # Get value at current position
                val = data_df.iloc[i, j]

                # Find extent of consecutive identical values
                start_j = j
                while j < n_cols and data_df.iloc[i, j] == val:
                    j += 1
                span = j - start_j

                # Store rectangle info: (row, col_start, col_span, row_span, value)
                rectangles_to_draw.append((i, start_j, span, 1, val))
    else:
        # No merging - one rectangle per cell
        for i in range(n_rows):
            for j in range(n_cols):
                val = data_df.iloc[i, j]
                rectangles_to_draw.append((i, j, 1, 1, val))

    # Draw the rectangles
    for rect_info in rectangles_to_draw:
        i, j, col_span, row_span, val = rect_info

        if horizontal:
            # Horizontal: blocks go left-to-right (x varies), rows stack vertically
            x_pos = j
            y_pos = n_rows - i - 1
            w = col_span
            h = row_span
        else:
            # Vertical: blocks stack top-to-bottom (y varies), x is constant
            x_pos = j
            y_pos = i
            w = col_span
            h = row_span

        # Store original center position for text (before inset)
        text_x = x_pos + w / 2
        text_y = y_pos + h / 2

        # Apply inset so linewidth grows inward, not outward
        x_pos += inset_x
        y_pos += inset_y
        w -= 2 * inset_x
        h -= 2 * inset_y

        # Get color for this block
        if isinstance(color_values[0], list):
            color = color_values[i][j]
        else:
            # For merged blocks, use color from first cell
            color = color_values[j] if horizontal else color_values[i]

        # Create single rectangle - apply_transparency will handle the double-layer effect
        edge_c = edgecolor if edgecolor else color
        rect = Rectangle(
            (x_pos, y_pos), w, h,
            facecolor=color,
            edgecolor=edge_c,
            linewidth=linewidth,
            zorder=1
        )
        ax.add_patch(rect)
        patches.append(rect)

        # Add text overlay if requested
        if text:
            text_kwargs = text_kws or {}
            text_defaults = {
                'fontsize': resolve_param("font.size"),
                'ha': 'center',
                'va': 'center',
                'color': 'black',
                'weight': 'normal',
                'rotation': 90 if not horizontal else 0  # Rotate text for vertical blocks
            }
            text_defaults.update(text_kwargs)

            # Use original center position (not inset position) for text alignment
            ax.text(
                text_x, text_y, str(val),
                **text_defaults,
                zorder=3
            )

    # Apply transparency using publiplots utility
    apply_transparency(patches, face_alpha=alpha, edge_alpha=1.0)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Store legend metadata on axes for complex_heatmap reconciliation
    if legend and len(patches) > 0:
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

            # Store for later (use first patch from local patches list)
            if not hasattr(patches[0], '_legend_data'):
                patches[0]._legend_data = {}
            patches[0]._legend_data['hue'] = {
                'handles': handles,
                'label': label
            }
        else:
            # Continuous colorbar
            mappable = ScalarMappable(norm=norm, cmap=cmap_obj)

            # Store for later (use first patch from local patches list)
            if not hasattr(patches[0], '_legend_data'):
                patches[0]._legend_data = {}
            patches[0]._legend_data['hue'] = {
                'type': 'colorbar',
                'mappable': mappable,
                'label': label
            }

    return fig, ax


def label(
    data: Union[pd.Series, pd.DataFrame, List, np.ndarray],
    # Position/orientation (optional when called from complex_heatmap)
    x: Optional[Union[str, bool]] = None,
    y: Optional[Union[str, bool]] = None,
    # Data handling
    order: Optional[List[str]] = None,
    merge: bool = True,
    # Text styling
    rotation: float = 0,
    fontsize: Optional[float] = None,
    ha: str = "center",
    va: str = "center",
    color: Optional[str] = None,
    fontweight: str = "normal",
    hue: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    # Arrow connection
    arrow: Optional[str] = None,
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
    data : Series, DataFrame, list, or array
        Data containing labels to display. Can be:
        - Series: Simple label list
        - DataFrame: Extract labels from column specified by x/y parameter
        - List/array: Simple label list
    x : str or bool, optional
        Column name (if data is DataFrame) or True for horizontal orientation.
        Use for top/bottom margins. Can be omitted when called from complex_heatmap.
    y : str or bool, optional
        Column name (if data is DataFrame) or True for vertical orientation.
        Use for left/right margins. Can be omitted when called from complex_heatmap.
    order : list of str, optional
        Order of positions (passed automatically by complex_heatmap builder).
    merge : bool, default=True
        If True, merge consecutive identical labels (only show once for the span).
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
        Only used when hue is None.
    fontweight : str, default='normal'
        Font weight: 'normal', 'bold', 'light', 'heavy'.
    hue : str, optional
        Column name in DataFrame to use for coloring labels.
        Colors both text and arrow. If provided, overrides 'color' parameter.
    palette : str, dict, or list, optional
        Color palette for hue values (used when hue is provided):
        - str: Palette name (e.g., 'Set2', 'pastel')
        - dict: Mapping from hue values to colors
        - list: List of colors
        If None and hue is provided, uses default palette.
    arrow : str, optional
        Direction of connecting arrows from labels to positions.
        Options: 'up', 'down', 'left', 'right', or None (no arrows).
        For horizontal labels (top/bottom), use 'up' or 'down'.
        For vertical labels (left/right), use 'left' or 'right'.
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
    Simple horizontal labels (for top/bottom margins):

    >>> gene_names = ['BRCA1', 'TP53', 'EGFR', 'MYC']
    >>> fig, ax = pp.label(gene_names, x=True, rotation=45, ha='right')

    Vertical labels with arrows (for left/right margins):

    >>> sample_ids = ['S001', 'S002', 'S003']
    >>> fig, ax = pp.label(sample_ids, y=True,
    ...                     arrow='left', arrow_kws={'arrowstyle': '->'})

    Custom styling:

    >>> categories = ['Group A', 'Group B', 'Group C']
    >>> fig, ax = pp.label(categories, x=True, fontsize=12, fontweight='bold',
    ...                     color='darkblue', rotation=90)

    With hue coloring from DataFrame:

    >>> metadata = pd.DataFrame({
    ...     'gene': ['BRCA1', 'TP53', 'EGFR'],
    ...     'category': ['Oncogene', 'Tumor suppressor', 'Oncogene']
    ... })
    >>> fig, ax = pp.label(metadata, y='gene', hue='category',
    ...                     palette={'Oncogene': 'red', 'Tumor suppressor': 'blue'},
    ...                     arrow='left')

    Notes
    -----
    - Designed for complex_heatmap margins but works standalone
    - Arrow connections useful for sparse labeling (not every position)
    - Rotation is applied after text is positioned
    - For dense labels, consider using pp.ticklabels instead
    """
    # Resolve defaults
    fontsize = resolve_param("font.size", fontsize)
    if color is None and hue is None:
        color = resolve_param("color", color)

    # Extract hue data if provided (before preparing annotation data)
    hue_data = None
    original_data = data
    if hue is not None and isinstance(data, pd.DataFrame):
        if hue in data.columns:
            hue_data = data[hue]
        else:
            raise ValueError(f"Hue column '{hue}' not found in DataFrame")

    # Prepare data using helper function
    data_df, horizontal, n_rows, n_cols = _prepare_annotation_data(data, x, y, order)

    # Create figure if needed
    if ax is None:
        figsize = (6, 0.8) if horizontal else (0.8, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Process hue and palette for color mapping (following publiplots pattern)
    color_map = None
    if hue is not None and hue_data is not None:
        # Get unique label values
        unique_labels = pd.unique(data_df.values.ravel())

        # Map labels to hue values
        # Need to align labels with their corresponding hue values
        label_to_hue = {}
        if isinstance(original_data, pd.DataFrame):
            # Create mapping from labels to hue values
            for label in unique_labels:
                # Find hue value for this label
                # Handle both cases: when label column is x/y, or when it's a separate column
                if isinstance(x, str) and x in original_data.columns:
                    # x is the label column
                    rows = original_data[original_data[x] == label]
                    if len(rows) > 0:
                        label_to_hue[label] = rows[hue].iloc[0]
                elif isinstance(y, str) and y in original_data.columns:
                    # y is the label column
                    rows = original_data[original_data[y] == label]
                    if len(rows) > 0:
                        label_to_hue[label] = rows[hue].iloc[0]

        # Get unique hue values
        unique_hue_values = pd.unique(pd.Series(list(label_to_hue.values()))) if label_to_hue else unique_labels

        # Resolve palette to color mapping
        if palette is None:
            palette = 'pastel'  # Default palette

        hue_color_map = resolve_palette_map(
            values=unique_hue_values,
            palette=palette
        )

        # Create label to color mapping
        if label_to_hue:
            color_map = {label: hue_color_map[hue_val] for label, hue_val in label_to_hue.items()}
        else:
            color_map = hue_color_map

    # Build list of labels to draw (with merging if requested)
    labels_to_draw = []

    if merge:
        # Merge consecutive identical values - only label once for the span
        for i in range(n_rows):
            j = 0
            while j < n_cols:
                # Get value at current position
                val = data_df.iloc[i, j]

                # Find extent of consecutive identical values
                start_j = j
                while j < n_cols and data_df.iloc[i, j] == val:
                    j += 1
                span = j - start_j

                # Store label info: (row, col_start, col_span, row_span, value)
                # Label will be centered on the span
                labels_to_draw.append((i, start_j, span, 1, val))
    else:
        # No merging - one label per cell
        for i in range(n_rows):
            for j in range(n_cols):
                val = data_df.iloc[i, j]
                labels_to_draw.append((i, j, 1, 1, val))

    # Position labels (with or without arrows)
    # Arrow takes ALL remaining space - text is at opposite end
    total_space = offset + 0.5

    for label_info in labels_to_draw:
        i, j, col_span, row_span, lbl = label_info

        # Get color for this label
        label_color = color
        if color_map is not None and lbl in color_map:
            label_color = color_map[lbl]
        elif label_color is None:
            label_color = 'black'

        # Calculate target position (where arrow points to)
        if horizontal:
            target_x = j + col_span / 2
            target_y = 0  # Points to heatmap edge
        else:
            target_x = 0  # Points to heatmap edge
            target_y = i + row_span / 2

        if arrow:
            # Use annotate() like pyComplexHeatmap for proper arrow connections
            # Following their exact coordinate system and positioning logic

            # Calculate arrow height in pixels (following pyComplexHeatmap)
            bbox = ax.get_position()
            if horizontal:
                arrow_height_inches = (offset + 0.5) * bbox.height * fig.get_figheight()
            else:
                arrow_height_inches = (offset + 0.5) * bbox.width * fig.get_figwidth()
            arrow_height_pixels = arrow_height_inches * fig.dpi

            # Calculate arm height for arc connectionstyle (20% of arrow height)
            arm_height_pixels = arrow_height_pixels * 0.2

            # Position text and arrow following pyComplexHeatmap's exact logic
            if arrow == 'down':
                # Horizontal annotation, arrow points DOWN (toward heatmap)
                # xy: bottom edge near heatmap (y=0), xytext: above it (positive offset)
                xy = (target_x, 0)
                xytext = (0, arrow_height_pixels)
                xycoords = ax.get_xaxis_transform()
                textcoords = "offset pixels"
                angleA = 180 + rotation
                angleB = -90
                relpos = (0.5, 0.0)  # Bottom center of text box
                # Adjust alignment for rotation
                if rotation == -90:
                    text_ha = 'left'
                    text_va = 'center'
                elif rotation == 90:
                    text_ha = 'right'
                    text_va = 'center'
                else:
                    text_ha = ha if ha != 'center' else 'center'
                    text_va = va if va != 'center' else 'top'
            elif arrow == 'up':
                # Horizontal annotation, arrow points UP (away from heatmap)
                # xy: top edge far from heatmap (y=1), xytext: below it (negative offset)
                xy = (target_x, 1)
                xytext = (0, -arrow_height_pixels)
                xycoords = ax.get_xaxis_transform()
                textcoords = "offset pixels"
                angleA = rotation - 180
                angleB = 90
                relpos = (0.5, 1.0)  # Top center of text box
                # Adjust alignment for rotation
                if rotation == 90:
                    text_ha = 'left'
                    text_va = 'center'
                elif rotation == -90:
                    text_ha = 'right'
                    text_va = 'center'
                else:
                    text_ha = ha if ha != 'center' else 'center'
                    text_va = va if va != 'center' else 'bottom'
            elif arrow == 'left':
                # Vertical annotation, arrow points LEFT (away from heatmap)
                # xy: left edge far from heatmap (x=0), xytext: right of it (positive offset)
                xy = (0, target_y)
                xytext = (arrow_height_pixels, 0)
                xycoords = ax.get_yaxis_transform()
                textcoords = "offset pixels"
                angleA = rotation
                angleB = -180
                relpos = (0.0, 0.5)  # Left center of text box
                text_ha = ha if ha != 'center' else 'right'
                text_va = va if va != 'center' else 'center'
            elif arrow == 'right':
                # Vertical annotation, arrow points RIGHT (toward heatmap)
                # xy: right edge near heatmap (x=1), xytext: left of it (negative offset)
                xy = (1, target_y)
                xytext = (-arrow_height_pixels, 0)
                xycoords = ax.get_yaxis_transform()
                textcoords = "offset pixels"
                angleA = rotation - 180
                angleB = 0
                relpos = (1.0, 0.5)  # Right center of text box
                text_ha = ha if ha != 'center' else 'left'
                text_va = va if va != 'center' else 'center'
            else:
                # Default to down
                xy = (target_x, 0)
                xytext = (0, arrow_height_pixels)
                xycoords = ax.get_xaxis_transform()
                textcoords = "offset pixels"
                angleA = 180 + rotation
                angleB = -90
                text_ha = ha if ha != 'center' else 'center'
                text_va = va if va != 'center' else 'top'
                relpos = (0.5, 0.0)

            # Setup arrow properties with arc connectionstyle
            arrow_kwargs = arrow_kws or {}

            # Extract arrow-specific parameters
            connectionstyle = arrow_kwargs.pop('connectionstyle', None)
            rad = arrow_kwargs.pop('rad', 2)
            arrow_linewidth = arrow_kwargs.pop('linewidth', None)
            arrow_linewidth = resolve_param("lines.linewidth", arrow_linewidth)

            if connectionstyle is None:
                # Use pyComplexHeatmap-style arc connection
                connectionstyle = f"arc,angleA={angleA},angleB={angleB},armA={arm_height_pixels},armB={arm_height_pixels},rad={rad}"

            arrowprops = {
                'arrowstyle': '-',
                'color': label_color,
                'linewidth': arrow_linewidth,
                'shrinkA': 1,  # 1 point shrink from text (like pyComplexHeatmap)
                'shrinkB': 1,  # 1 point shrink from target
                'connectionstyle': connectionstyle,
                'relpos': relpos,
            }
            arrowprops.update(arrow_kwargs)

            # Draw text with arrow using annotate
            text_kwargs = kwargs.copy()
            text_kwargs.update({
                'fontsize': fontsize,
                'ha': text_ha,
                'va': text_va,
                'color': label_color,
                'weight': fontweight,
                'rotation': rotation,
                'rotation_mode': 'anchor',
            })

            ax.annotate(
                str(lbl),
                xy=xy,
                xytext=xytext,
                xycoords=xycoords,
                textcoords=textcoords,
                arrowprops=arrowprops,
                **text_kwargs,
                zorder=2
            )
        else:
            # No arrow - draw text only
            if horizontal:
                x_pos = target_x
                y_pos = offset
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
                x_pos = offset
                y_pos = target_y
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
                'color': label_color,
                'weight': fontweight,
                'rotation': rotation,
            })

            ax.text(x_pos, y_pos, str(lbl), **text_kwargs, zorder=2)

    # Set axis limits
    if horizontal:
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, offset + 0.5)
        ax.invert_yaxis()
    else:
        ax.set_xlim(0, offset + 0.5)
        ax.set_ylim(0, n_rows)
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
