"""
Unified legend system for complex heatmaps.

This module provides functionality to collect and display legends from all
margin plots and the main heatmap in a single unified legend panel.
"""

from typing import Optional, Union, List, Tuple, Dict, Any
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from publiplots.utils.legend import LegendBuilder, get_legend_handler_map


def legend(
    ax: Optional[Axes] = None,
    collect_from: Union[str, List[str]] = "all",
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create a unified legend collecting from all axes in a complex heatmap.

    This function is designed to be used with ComplexHeatmapBuilder's .add_*() methods
    to create a single legend panel that collects legend entries from all margin plots
    and the main heatmap.

    Parameters
    ----------
    ax : Axes, optional
        The axes to draw the legend in. This will be provided automatically by
        ComplexHeatmapBuilder when used with .add_*() methods.
    collect_from : str or list of str, default="all"
        Which axes to collect legends from:
        - "all": Collect from all axes including main heatmap
        - "main": Only collect from main heatmap
        - list of positions: e.g., ["main", "top", "left"]
    **kwargs
        Additional parameters passed to LegendBuilder:
        - gap: Vertical spacing between legend elements (mm)
        - x_offset: Horizontal offset from axes edge (mm)
        - vpad: Vertical padding from top (mm)
        - Other kwargs passed to individual legend additions

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    ax : Axes
        The axes containing the unified legend.

    Examples
    --------
    Basic usage with complex heatmap:

    >>> fig, axes = (
    ...     pp.complex_heatmap(data, row_cluster=True)
    ...     .add_top(pp.barplot, data=counts)
    ...     .add_left(pp.dendrogram)
    ...     .add_right(pp.legend, width='auto')  # Unified legend
    ...     .build()
    ... )

    Collect only from specific positions:

    >>> fig, axes = (
    ...     pp.complex_heatmap(data)
    ...     .add_top(pp.barplot, data=counts)
    ...     .add_right(pp.legend, width='auto', collect_from=["main", "top"])
    ...     .build()
    ... )

    Notes
    -----
    The function automatically:
    - Collects legends from axes with `._legend_builder` attribute
    - Skips axes where legend=False was set
    - Keeps both entries if labels conflict (different artists)
    - Maintains the order plots were added
    """
    if ax is None:
        raise ValueError("ax parameter is required for unified legend")

    # This function will be called by ComplexHeatmapBuilder._draw_margin_plot
    # At that point, ax is the legend panel axes and we need to access the parent figure
    # and all other axes to collect their legends

    fig = ax.get_figure()

    # Get all axes in the figure
    all_axes = fig.get_axes()

    # Filter to get the axes we want to collect from
    # The builder should have stored metadata on axes to identify them
    axes_to_collect = []

    if collect_from == "all":
        # Collect from all axes that have _legend_builder
        for a in all_axes:
            if hasattr(a, '_legend_builder') and a._legend_builder is not None:
                axes_to_collect.append(a)
    elif collect_from == "main":
        # Only collect from main heatmap
        for a in all_axes:
            if hasattr(a, '_is_main_heatmap') and a._is_main_heatmap:
                axes_to_collect.append(a)
    elif isinstance(collect_from, list):
        # Collect from specific positions
        for a in all_axes:
            if hasattr(a, '_margin_position') and a._margin_position in collect_from:
                axes_to_collect.append(a)
            if "main" in collect_from and hasattr(a, '_is_main_heatmap') and a._is_main_heatmap:
                axes_to_collect.append(a)

    # Initialize the unified legend builder on the legend panel
    gap = kwargs.pop('gap', 2)
    x_offset = kwargs.pop('x_offset', 0)  # Start from left edge of legend panel
    vpad = kwargs.pop('vpad', 5)

    # Create legend builder for this axes
    builder = LegendBuilder(
        ax,
        x_offset=x_offset,
        gap=gap,
        vpad=vpad,
    )

    # Collect all legend entries from the collected axes
    # Track labels to handle conflicts
    collected_entries = []  # List of (label, handles, source_ax, legend_data)
    labels_seen = {}  # label -> list of (handles, source_ax)

    for source_ax in axes_to_collect:
        if not hasattr(source_ax, '_legend_builder'):
            continue

        source_builder = source_ax._legend_builder

        # Iterate through the stored elements in the source builder
        for element_type, element in source_builder.elements:
            if element_type == "legend":
                # Extract handles and labels from the legend
                handles = element.legendHandles
                labels = [h.get_label() for h in handles]
                title = element.get_title().get_text() if element.get_title() else ""

                # Check for label conflicts
                for label, handle in zip(labels, handles):
                    if label in labels_seen:
                        # Check if artists are different
                        existing_handles = [h for h, _ in labels_seen[label]]
                        # Keep both if different artists
                        # For now, just keep both (user can control via legend=False)
                        labels_seen[label].append((handle, source_ax))
                    else:
                        labels_seen[label] = [(handle, source_ax)]

                # Store the entire legend for reconstruction
                collected_entries.append((title, handles, labels, source_ax))

    # Reconstruct legends in the unified panel
    # Maintain the order they were added (order of axes_to_collect)
    for title, handles, labels, source_ax in collected_entries:
        builder.add_legend(
            handles=handles,
            label=title,
            **kwargs
        )

    # Hide axes frame and ticks for clean legend panel
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    return fig, ax


def calculate_legend_space(
    axes_list: List[Axes],
    orientation: str = "vertical",
    **kwargs
) -> float:
    """
    Calculate the space needed for a unified legend panel.

    Parameters
    ----------
    axes_list : list of Axes
        List of axes to collect legends from.
    orientation : str, default="vertical"
        Orientation of the legend panel.
    **kwargs
        Additional parameters for legend builder.

    Returns
    -------
    float
        Required space in millimeters.
    """
    # TODO: Implement size estimation based on collected legends
    # For now, return a default size
    return 30  # mm


__all__ = [
    'legend',
    'calculate_legend_space',
]
