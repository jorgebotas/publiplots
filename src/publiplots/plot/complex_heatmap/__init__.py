"""
Complex heatmap module for publiplots.

Provides advanced heatmap functionality including:
- Hierarchical clustering with dendrograms
- Margin plots (barplots, annotations, etc.)
- Method chaining builder API
"""

from .builder import complex_heatmap, ComplexHeatmapBuilder
from .dendrogram import dendrogram, cluster_data
from .ticklabels import ticklabels
from .legend import legend

__all__ = [
    'complex_heatmap',
    'ComplexHeatmapBuilder',
    'dendrogram',
    'ticklabels',
    'legend',
    'cluster_data',
]
