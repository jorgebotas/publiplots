"""
Swarm plot functions for publiplots.

This module provides publication-ready swarm plot visualizations with
transparent fill and opaque edges.
"""

from typing import Optional, List, Dict, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils.transparency import ArtistTracker
from publiplots.utils.legend import create_legend_handles
from publiplots.utils.legend import legend as legend_fn
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)


def swarmplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    dodge: bool = False,
    orient: Optional[str] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    size: float = 5,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    hue_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    alpha: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs
) -> Axes:
    """
    Create a publication-ready swarm plot.

    This function creates swarm plots with transparent fill and opaque edges,
    following the publiplots visual style.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x : str, optional
        Column name for x-axis variable.
    y : str, optional
        Column name for y-axis variable.
    hue : str, optional
        Column name for color grouping.
    order : list, optional
        Order for the categorical levels.
    hue_order : list, optional
        Order for the hue levels.
    dodge : bool, default=False
        Whether to separate points by hue along the categorical axis.
    orient : str, optional
        Orientation of the plot ('v' or 'h').
    color : str, optional
        Fixed color for all points (only used when hue is None).
    palette : str, dict, or list, optional
        Color palette for hue grouping.
    size : float, default=5
        Size of the markers.
    edgecolor : str, optional
        Color for marker edges. If None, uses face color.
    linewidth : float, optional
        Width of marker edges.
    hue_norm : tuple or Normalize, optional
        Normalization for continuous hue variable.
    alpha : float, optional
        Transparency of marker fill (0-1).
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label.
    ylabel : str, default=""
        Y-axis label.
    legend : bool, default=True
        Whether to show the legend.
    legend_kws : dict, optional
        Additional keyword arguments for legend.
    **kwargs
        Additional keyword arguments passed to seaborn.swarmplot.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple swarm plot:

    >>> import publiplots as pp
    >>> ax = pp.swarmplot(data=df, x="category", y="value")

    Swarm plot with hue grouping:

    >>> ax = pp.swarmplot(
    ...     data=df, x="category", y="value", hue="group"
    ... )
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Create figure via pp.subplots to install SubplotsAutoLayout; users who
    # want custom dimensions should compose with pp.subplots(axes_size=...)
    # before calling and pass ax=.
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Resolve palette
    if hue is not None:
        palette = resolve_palette_map(
            values=data[hue].unique(),
            palette=palette,
        )

    # Prepare kwargs for seaborn swarmplot
    swarmplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "dodge": dodge,
        "orient": orient,
        "color": color if hue is None else None,
        "palette": palette if hue else None,
        "size": size,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "hue_norm": hue_norm,
        "ax": ax,
        "legend": False,  # Handle legend ourselves
    }

    # Merge with user-provided kwargs
    swarmplot_kwargs.update(kwargs)

    # Track artists before plotting
    tracker = ArtistTracker(ax)

    # Create swarmplot
    sns.swarmplot(**swarmplot_kwargs)

    # Set edge colors if not specified
    if edgecolor is None:
        for collection in tracker.get_new_collections():
            collection.set_edgecolors(collection.get_facecolors())

    # Apply transparency to new collections only
    tracker.apply_transparency(on="collections", face_alpha=alpha)

    # Stash legend entry (per-kind) and optionally render per-axis legend.
    # legend may be bool or dict[str, bool]; False short-circuits both stash
    # and render, True stashes/renders everything, and a dict filters per kind.
    if hue is not None:
        _legend(
            ax=ax,
            hue=hue,
            color=color,
            edgecolor=edgecolor,
            palette=palette,
            hue_norm=hue_norm,
            alpha=alpha,
            linewidth=linewidth,
            kwargs=legend_kws,
            legend=legend,
        )

    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax


def _legend(
    ax: Axes,
    hue: Optional[str],
    color: Optional[str],
    edgecolor: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    hue_norm: Optional[Normalize] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    kwargs: Optional[Dict] = None,
    legend: Union[bool, Dict] = True,
) -> None:
    """
    Stash LegendEntry objects for swarm plot and optionally render per-axis legend.

    Parameters
    ----------
    legend : bool or dict, default=True
        - ``True``  -> stash all kinds and render per-axis (unless claimed by a group).
        - ``False`` -> stash nothing and render nothing.
        - ``dict``  -> per-kind flags (missing keys default to True).
    """
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)

    kwargs = kwargs or {}
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="circle")

    flags = resolve_legend_flags(legend)

    # Stash hue entry
    if hue is not None and flags["hue"]:
        hue_label = kwargs.pop("hue_label", hue)
        if isinstance(palette, dict):  # categorical
            hue_labels = list(palette.keys())
            hue_handles = create_legend_handles(
                labels=hue_labels,
                colors=list(palette.values()),
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                **handle_kwargs
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=hue_handles,
                    labels=hue_labels,
                ),
            )
        else:
            # continuous -> colorbar
            mappable = ScalarMappable(norm=hue_norm, cmap=palette)
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=[mappable],
                    labels=[],
                ),
            )

    # Render per-axis legend for entries not claimed by a figure-level group.
    # legend=False short-circuits (all flags False, nothing stashed, nothing to render).
    if legend is False:
        return

    fig = ax.get_figure()
    entries_to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e)
    ]
    if not entries_to_render:
        return

    builder = legend_fn(ax=ax, auto=False)
    for entry in entries_to_render:
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            builder.add_colorbar(
                mappable=entry.handles[0],
                label=entry.name,
            )
        else:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )
