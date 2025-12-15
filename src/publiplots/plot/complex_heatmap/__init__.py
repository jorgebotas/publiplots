"""
Complex heatmap module for publiplots.

Provides advanced heatmap functionality including:
- Hierarchical clustering with dendrograms
- Margin plots (barplots, annotations, etc.)
- Method chaining builder API
- Annotation functions (block, label, spacer)
"""

from .builder import complex_heatmap, ComplexHeatmapBuilder
from .dendrogram import dendrogram, cluster_data
from .ticklabels import ticklabels
from .annotations import block, label, spacer

__all__ = [
    "complex_heatmap",
    "ComplexHeatmapBuilder",
    "dendrogram",
    "ticklabels",
    "cluster_data",
    "block",
    "label",
    "spacer",
]
