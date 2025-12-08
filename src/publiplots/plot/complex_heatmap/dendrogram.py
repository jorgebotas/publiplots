"""
Dendrogram and clustering utilities for complex heatmaps.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from publiplots.themes.rcparams import resolve_param


def cluster_data(
    matrix: pd.DataFrame,
    row_cluster: bool = True,
    col_cluster: bool = True,
    method: str = "ward",
    metric: str = "euclidean",
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Cluster rows and/or columns of a matrix.

    Parameters
    ----------
    matrix : DataFrame
        Data matrix to cluster.
    row_cluster : bool, default=True
        Whether to cluster rows.
    col_cluster : bool, default=True
        Whether to cluster columns.
    method : str, default="ward"
        Clustering linkage method (ward, single, complete, average).
    metric : str, default="euclidean"
        Distance metric for clustering.

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
    Draw a dendrogram for hierarchical clustering.

    Parameters
    ----------
    linkage : ndarray, optional
        Precomputed linkage matrix. If None, computed from data.
    data : DataFrame, optional
        Data to cluster (if linkage not provided).
    method : str, default="ward"
        Clustering linkage method (ward, single, complete, average).
    metric : str, default="euclidean"
        Distance metric for clustering.
    orientation : str, default="top"
        Dendrogram orientation: 'top', 'bottom', 'left', 'right'.
    color : str, optional
        Line color. Uses rcParams default if None.
    linewidth : float, optional
        Line width. Uses rcParams default if None.
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure and axes.
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
