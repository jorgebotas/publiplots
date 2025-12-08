# Complex Heatmap Implementation Plan

**Project**: PubliPlots Complex Heatmap System
**Inspired by**: [pyComplexHeatmap](https://github.com/DingWB/PyComplexHeatmap)
**Created**: 2025-12-08
**Status**: Planning Phase

---

## Overview

This document outlines the implementation plan for adding complex heatmap capabilities to PubliPlots, enabling publication-ready heatmaps with margin plots, annotations, and clustering support.

### Goals

1. Create `pp.heatmap()` for standard heatmaps (color-encoded and dot/bubble)
2. Create `pp.complex_heatmap()` builder for composable margin plots
3. Support annotations for row/column metadata visualization
4. Maintain PubliPlots' functional API style and publication-ready defaults

### Design Principles

- **Functional over Object-Oriented**: Keep consistent with PubliPlots' existing API
- **Composable**: Margin plots work with ANY plot function, not just built-in ones
- **Sensible Defaults**: Publication-ready output with minimal configuration
- **Format Flexible**: Support both wide-format (matrix) and long-format (tidy) data

---

## Stage 1: Core Heatmap Function

**File**: `publiplots/plot/heatmap.py`
**Function**: `pp.heatmap()`

### 1.1 Basic Heatmap (sns.heatmap wrapper)

#### Function Signature

```python
def heatmap(
    data: pd.DataFrame,
    # Long-format parameters (optional)
    x: Optional[str] = None,
    y: Optional[str] = None,
    value: Optional[str] = None,
    # Dot heatmap mode
    size: Optional[str] = None,
    # Color encoding
    hue: Optional[str] = None,  # Alternative to value for dot mode
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    # Annotations
    annot: bool = False,
    annot_kws: Optional[Dict] = None,
    fmt: str = ".2g",
    # Styling
    linewidths: float = 0,
    linecolor: str = "white",
    square: bool = False,
    # Size encoding (dot mode)
    sizes: Optional[Tuple[float, float]] = None,
    size_norm: Optional[Tuple[float, float]] = None,
    # Standard publiplots params
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    # Legend
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
```

#### Data Format Handling

```python
# Internal logic:
if x is not None and y is not None and value is not None:
    # Long-format -> pivot to wide
    matrix = data.pivot(index=y, columns=x, values=value)
    if size is not None:
        size_matrix = data.pivot(index=y, columns=x, values=size)
else:
    # Wide-format (DataFrame is the matrix)
    matrix = data
    size_matrix = None
```

#### Mode Detection

```python
# Automatic mode selection:
if size is not None:
    # DOT HEATMAP MODE
    # - Create categorical scatter grid
    # - Size encodes one variable, color encodes another
    _draw_dot_heatmap(...)
else:
    # STANDARD HEATMAP MODE
    # - Use sns.heatmap internally
    sns.heatmap(matrix, ax=ax, cmap=cmap, ...)
```

### 1.2 Dot Heatmap Implementation

When `size` is provided, create a bubble/dot heatmap:

```python
def _draw_dot_heatmap(
    ax: Axes,
    data: pd.DataFrame,  # Long format
    x: str,
    y: str,
    value: str,          # Color encoding
    size: str,           # Size encoding
    cmap: str,
    sizes: Tuple[float, float],
    ...
) -> None:
    """
    Draw dot heatmap using categorical scatter approach.

    Implementation:
    1. Create position mappings for x and y categories
    2. Normalize size values to sizes range
    3. Normalize color values to cmap
    4. Draw scatter points at grid intersections
    5. Apply publiplots double-layer styling (fill + edge)
    """
```

### 1.3 Tasks for Stage 1

- [ ] Create `publiplots/plot/heatmap.py` with basic structure
- [ ] Implement wide-format heatmap using `sns.heatmap`
- [ ] Add long-format support with pivot transformation
- [ ] Implement dot heatmap mode with size parameter
- [ ] Add color normalization and colorbar legend
- [ ] Add size legend for dot mode
- [ ] Apply publiplots styling (transparency, linewidth)
- [ ] Write unit tests for both modes
- [ ] Add example to documentation

---

## Stage 2: Complex Heatmap Builder

**File**: `publiplots/plot/heatmap.py` (extend)
**Function**: `pp.complex_heatmap()` returning `ComplexHeatmapBuilder`

### 2.1 Builder Class Design

```python
class ComplexHeatmapBuilder:
    """
    Builder for complex heatmaps with margin plots.

    Usage:
        fig, axes = (
            pp.complex_heatmap(data, x="gene", y="sample", value="expr")
            .add_top(pp.barplot, data=col_summary, x="sample", y="total", height=15)
            .add_left(pp.barplot, data=row_summary, x="mean", y="gene", width=15)
            .add_top(pp.dendrogram, height=10)  # Multiple top plots stack
            .add_right(custom_func, width=20)
            .build()
        )

    Returns:
        fig: matplotlib Figure
        axes: dict with keys 'main', 'top', 'bottom', 'left', 'right'
              Each margin key contains a list of axes if multiple plots
    """

    def __init__(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        value: Optional[str] = None,
        size: Optional[str] = None,
        # ... all heatmap parameters
        figsize: Optional[Tuple[float, float]] = None,
        # Gaps between main plot and margins
        hspace: float = 1.0,  # mm
        wspace: float = 1.0,  # mm
    ):
        self._heatmap_params = {...}
        self._margins = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
        }
        self._figsize = figsize
        self._hspace = hspace
        self._wspace = wspace
```

### 2.2 Add Margin Methods

```python
def add_top(
    self,
    func: Callable,
    height: float = 15,  # mm
    data: Optional[pd.DataFrame] = None,
    align: bool = True,  # Align x-axis with heatmap columns
    gap: float = 0,      # Additional gap from previous element
    **kwargs
) -> 'ComplexHeatmapBuilder':
    """
    Add a plot to the top margin.

    Parameters
    ----------
    func : callable
        Plot function (e.g., pp.barplot, pp.violinplot, custom function).
        Must accept `ax` parameter and return (fig, ax).
    height : float
        Height in millimeters.
    data : DataFrame, optional
        Data for the plot. If None, uses subset of main data.
    align : bool
        If True, share x-axis with heatmap for alignment.
    gap : float
        Additional gap (mm) between this and previous element.
    **kwargs
        Additional parameters passed to func.
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

def add_left(self, func, width=15, data=None, align=True, gap=0, **kwargs):
    """Add a plot to the left margin. Similar to add_top."""
    ...

def add_right(self, func, width=15, data=None, align=True, gap=0, **kwargs):
    """Add a plot to the right margin."""
    ...

def add_bottom(self, func, height=15, data=None, align=True, gap=0, **kwargs):
    """Add a plot to the bottom margin."""
    ...
```

### 2.3 Build Method - GridSpec Layout

```python
def build(self) -> Tuple[plt.Figure, Dict[str, Union[Axes, List[Axes]]]]:
    """
    Build the complex heatmap with all margin plots.

    Layout Strategy:
    1. Calculate total dimensions based on margin sizes
    2. Create GridSpec with appropriate ratios
    3. Create axes with shared x/y where alignment requested
    4. Draw main heatmap
    5. Draw margin plots in order
    6. Handle tick label visibility
    7. Collect and position legends
    """
    # Calculate layout
    n_top = len(self._margins['top'])
    n_bottom = len(self._margins['bottom'])
    n_left = len(self._margins['left'])
    n_right = len(self._margins['right'])

    n_rows = n_top + 1 + n_bottom
    n_cols = n_left + 1 + n_right

    # Calculate size ratios (convert mm to relative units)
    height_ratios = self._calculate_height_ratios()
    width_ratios = self._calculate_width_ratios()

    # Create figure and GridSpec
    fig = plt.figure(figsize=self._calculate_figsize())
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=self._hspace / 25.4 / fig.get_figheight(),  # Convert mm to figure fraction
        wspace=self._wspace / 25.4 / fig.get_figwidth(),
    )

    # Create main heatmap axes
    main_row = n_top
    main_col = n_left
    ax_main = fig.add_subplot(gs[main_row, main_col])

    # Draw main heatmap
    _draw_heatmap(ax=ax_main, **self._heatmap_params)

    # Create and draw margin plots
    axes = {'main': ax_main, 'top': [], 'bottom': [], 'left': [], 'right': []}

    # Top margins (from bottom to top, so index 0 is closest to heatmap)
    for i, margin in enumerate(self._margins['top']):
        row_idx = n_top - 1 - i
        ax = fig.add_subplot(
            gs[row_idx, main_col],
            sharex=ax_main if margin['align'] else None
        )
        margin['func'](data=margin['data'], ax=ax, **margin['kwargs'])
        if margin['align']:
            ax.tick_params(labelbottom=False)  # Hide x labels
        axes['top'].append(ax)

    # Similar for left, right, bottom...

    return fig, axes
```

### 2.4 Dendrogram Support

```python
def dendrogram(
    data: Optional[pd.DataFrame] = None,
    linkage: Optional[np.ndarray] = None,
    method: str = "ward",
    metric: str = "euclidean",
    orientation: str = "top",  # 'top', 'bottom', 'left', 'right'
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes, np.ndarray]:
    """
    Draw a dendrogram.

    Uses scipy.cluster.hierarchy for clustering and dendrogram calculation.

    Returns
    -------
    fig : Figure
    ax : Axes
    order : ndarray
        Leaf ordering from clustering (for reordering heatmap data)
    """
    from scipy.cluster.hierarchy import linkage as compute_linkage, dendrogram as draw_dendrogram

    if linkage is None and data is not None:
        linkage = compute_linkage(data, method=method, metric=metric)

    # Draw dendrogram
    dendro = draw_dendrogram(
        linkage,
        ax=ax,
        orientation=orientation,
        no_labels=True,
        color_threshold=0,
        above_threshold_color=color or resolve_param("color"),
        **kwargs
    )

    # Style axes (remove spines, ticks)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax, np.array(dendro['leaves'])
```

### 2.5 Clustering Integration

Add clustering parameters to `ComplexHeatmapBuilder`:

```python
class ComplexHeatmapBuilder:
    def __init__(
        self,
        ...,
        row_cluster: bool = False,
        col_cluster: bool = False,
        cluster_method: str = "ward",
        cluster_metric: str = "euclidean",
        row_dendrogram: bool = True,  # Show dendrogram if clustered
        col_dendrogram: bool = True,
        dendrogram_size: float = 10,  # mm
    ):
        ...

    def build(self):
        # If clustering enabled, compute linkage and reorder data
        if self._col_cluster:
            col_linkage = compute_linkage(matrix.T, ...)
            col_order = dendrogram(..., no_plot=True)['leaves']
            matrix = matrix.iloc[:, col_order]

            if self._col_dendrogram:
                # Auto-add dendrogram to top
                self._margins['top'].insert(0, {
                    'func': dendrogram,
                    'size': self._dendrogram_size,
                    'kwargs': {'linkage': col_linkage, 'orientation': 'top'},
                    'align': True,
                })

        # Similar for row clustering...
```

### 2.6 Tasks for Stage 2

- [ ] Create `ComplexHeatmapBuilder` class skeleton
- [ ] Implement `add_top()`, `add_left()`, `add_right()`, `add_bottom()` methods
- [ ] Implement GridSpec layout calculation in `build()`
- [ ] Implement axis sharing for alignment
- [ ] Create `pp.dendrogram()` function using scipy
- [ ] Add clustering parameters and auto-dendrogram
- [ ] Handle tick label visibility for shared axes
- [ ] Test with existing publiplots functions (barplot, violinplot, etc.)
- [ ] Test with custom user functions
- [ ] Verify alignment with different data sizes
- [ ] Add comprehensive examples

---

## Stage 3: Annotation System

**File**: `publiplots/plot/heatmap.py` (extend) or `publiplots/plot/annotations.py`

### 3.1 Annotation Types

Following pyComplexHeatmap's approach but with functional API:

```python
# Simple annotation strip (categorical -> colors)
def anno_simple(
    data: pd.Series,
    palette: Optional[Dict] = None,
    cmap: Optional[str] = None,
    ax: Optional[Axes] = None,
    orientation: str = "horizontal",  # or "vertical"
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Simple color strip annotation.

    For categorical data: discrete color blocks
    For continuous data: mini-heatmap strip
    """

# Bar annotation
def anno_bar(
    data: pd.Series,
    color: Optional[str] = None,
    orientation: str = "horizontal",
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """Bar chart annotation."""

# Label annotation
def anno_label(
    data: pd.Series,
    ax: Optional[Axes] = None,
    rotation: float = 0,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """Text label annotation."""

# Boxplot annotation
def anno_boxplot(
    data: pd.DataFrame,
    ax: Optional[Axes] = None,
    orientation: str = "horizontal",
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """Boxplot annotation showing distribution per row/column."""
```

### 3.2 Annotation Container

For convenience, support DataFrame-based automatic annotations:

```python
def annotation(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    colors: Optional[Dict[str, Dict]] = None,
    ax: Optional[Axes] = None,
    orientation: str = "horizontal",
    gap: float = 0.5,  # mm between annotation tracks
    **kwargs
) -> Tuple[plt.Figure, List[Axes]]:
    """
    Create multiple annotation tracks from a DataFrame.

    Each column becomes an annotation strip. Categorical columns
    get discrete colors, numeric columns get continuous colormaps.

    Usage with ComplexHeatmapBuilder:
        builder.add_top(pp.annotation, data=metadata_df, height=10)
    """
```

### 3.3 Integration with ComplexHeatmapBuilder

```python
# Convenience parameters for common annotation patterns
class ComplexHeatmapBuilder:
    def __init__(
        self,
        ...,
        col_annotation: Optional[pd.DataFrame] = None,
        row_annotation: Optional[pd.DataFrame] = None,
        col_annotation_kws: Optional[Dict] = None,
        row_annotation_kws: Optional[Dict] = None,
    ):
        # Auto-add annotations if provided
        if col_annotation is not None:
            self.add_top(
                annotation,
                data=col_annotation,
                height=len(col_annotation.columns) * 5,  # 5mm per track
                **(col_annotation_kws or {})
            )
```

### 3.4 Tasks for Stage 3

- [ ] Implement `anno_simple()` for color strips
- [ ] Implement `anno_bar()` for bar annotations
- [ ] Implement `anno_label()` for text labels
- [ ] Implement `anno_boxplot()` for distribution annotations
- [ ] Create `annotation()` container function
- [ ] Add convenience parameters to ComplexHeatmapBuilder
- [ ] Integrate annotation legends with LegendBuilder
- [ ] Handle mixed categorical/continuous annotations
- [ ] Test annotation alignment with clustered heatmaps
- [ ] Add examples for genomics use cases (gene expression, mutations)

---

## Stage 4: Polish and Documentation

### 4.1 Legend System Integration

- [ ] Extend `LegendBuilder` to collect legends from all margin plots
- [ ] Add `legend_loc` parameter to `ComplexHeatmapBuilder.build()`
- [ ] Support automatic legend positioning outside the plot area
- [ ] Handle overlapping legends from multiple annotations

### 4.2 Styling Consistency

- [ ] Ensure all annotations use publiplots' double-layer styling
- [ ] Apply consistent color resolution via `resolve_palette_map()`
- [ ] Use `resolve_param()` for all default values
- [ ] Verify publication-ready output at common figure sizes

### 4.3 Performance Optimization

- [ ] Add `rasterized` parameter for large heatmaps (>5000 cells)
- [ ] Optimize clustering for large datasets (consider fastcluster)
- [ ] Profile and optimize GridSpec layout calculation

### 4.4 Documentation

- [ ] Write docstrings for all public functions
- [ ] Create tutorial notebook for basic heatmaps
- [ ] Create tutorial notebook for complex heatmaps with margins
- [ ] Create tutorial notebook for genomics use cases
- [ ] Add API reference to docs
- [ ] Create visual gallery of examples

### 4.5 Testing

- [ ] Unit tests for heatmap data format conversion
- [ ] Unit tests for dot heatmap mode
- [ ] Unit tests for ComplexHeatmapBuilder
- [ ] Integration tests for margin plot alignment
- [ ] Visual regression tests for key examples
- [ ] Test edge cases (empty data, single row/column, etc.)

---

## File Structure

```
publiplots/
└── src/publiplots/
    └── plot/
        ├── __init__.py          # Add heatmap exports
        ├── heatmap.py           # NEW: Main heatmap module
        │   ├── heatmap()        # Simple heatmap function
        │   ├── complex_heatmap() # Returns ComplexHeatmapBuilder
        │   ├── ComplexHeatmapBuilder  # Builder class
        │   ├── dendrogram()     # Dendrogram plotting
        │   └── _draw_dot_heatmap()  # Internal dot heatmap
        └── annotations.py       # NEW: Annotation functions (Stage 3)
            ├── anno_simple()
            ├── anno_bar()
            ├── anno_label()
            ├── anno_boxplot()
            └── annotation()     # Container function
```

---

## API Summary

### Stage 1: Basic Heatmap

```python
# Standard color heatmap (wide format)
fig, ax = pp.heatmap(matrix_df, cmap="viridis", annot=True)

# Standard color heatmap (long format)
fig, ax = pp.heatmap(long_df, x="sample", y="gene", value="expression")

# Dot/bubble heatmap
fig, ax = pp.heatmap(long_df, x="sample", y="gene", value="expression", size="pvalue")
```

### Stage 2: Complex Heatmap with Margins

```python
fig, axes = (
    pp.complex_heatmap(data, x="sample", y="gene", value="expression",
                       row_cluster=True, col_cluster=True)
    .add_top(pp.barplot, data=totals_df, x="sample", y="total", height=15)
    .add_left(pp.barplot, data=means_df, x="mean", y="gene", width=15)
    .add_right(pp.violinplot, data=dist_df, x="gene", y="value", width=20)
    .build()
)

# Access individual axes
axes['main']      # Main heatmap
axes['top'][0]    # First top margin plot
axes['left'][0]   # First left margin plot
```

### Stage 3: With Annotations

```python
fig, axes = (
    pp.complex_heatmap(data, x="sample", y="gene", value="expression",
                       col_annotation=sample_metadata,
                       row_annotation=gene_metadata)
    .add_top(pp.dendrogram, height=10)
    .build()
)

# Or explicit annotation control
fig, axes = (
    pp.complex_heatmap(data, x="sample", y="gene", value="expression")
    .add_top(pp.annotation, data=sample_metadata,
             columns=["cell_type", "treatment"], height=10)
    .add_left(pp.anno_bar, data=gene_df["mean"], width=15)
    .build()
)
```

---

## Dependencies

### Required (existing)
- matplotlib
- seaborn
- pandas
- numpy

### New Dependencies
- scipy (for hierarchical clustering) - likely already a dependency via seaborn

### Optional
- fastcluster (for optimized clustering on large datasets)

---

## References

- [pyComplexHeatmap](https://github.com/DingWB/PyComplexHeatmap) - Primary inspiration
- [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) - Reference for basic clustering
- [matplotlib GridSpec](https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html) - Layout system

---

## Notes

### Why not use sns.clustermap?

`sns.clustermap` returns a `ClusterGrid` object with limited flexibility:
- Hard to add custom margin plots
- Dendrogram styling is limited
- Doesn't integrate with our builder pattern

Instead, we use `sns.heatmap` for the core heatmap and implement our own:
- Clustering via `scipy.cluster.hierarchy` (same as pyComplexHeatmap)
- Dendrogram drawing via scipy with custom styling
- GridSpec layout for full control over margins

This gives us the flexibility to support ANY plot function in margins, not just predefined annotation types.

### Alignment Strategy

For margin plots to align with heatmap rows/columns:
1. Use `sharex`/`sharey` when creating axes
2. Margin plot data must have matching index/categories
3. Hide tick labels on shared axes to avoid duplication

The builder tracks alignment requirements and handles this automatically.
