"""
Hexbin plot functions for publiplots.

Bivariate 2D-density visualization via hexagonal binning. Each hex is
colored by the count of points falling in it, or by a reduced statistic
of a third column ``C`` (mean / median / etc.). The color legend is a
continuous-hue colorbar rendered through the standard publiplots legend
reactor — so ``pp.legend(side='right')``, ``legend_kws={'inside': True}``,
and figure-anchored bands all work without any plot-specific legend code.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from publiplots.themes.colors import resolve_continuous_cmap
from publiplots.themes.rcparams import resolve_param
from publiplots.utils.legend_entries import resolve_legend_flags
from publiplots.utils.plot_legend import render_entries, stash_continuous_hue


def hexbinplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    C: Optional[str] = None,
    reduce_C_function: Callable = np.mean,
    gridsize: Union[int, Tuple[int, int]] = 30,
    bins: Optional[Union[str, int, Sequence]] = None,
    mincnt: int = 1,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    alpha: float = 1.0,
    extent: Optional[Tuple[float, float, float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs,
) -> Axes:
    """
    Create a publication-ready hexagonal-binning density plot.

    Aggregates ``(x, y)`` point clouds into a hexagonal grid, coloring
    each hex by either its count or a reduced statistic (mean / median /
    etc.) of an auxiliary column ``C``. The color legend is a continuous
    colorbar routed through the standard publiplots legend reactor.

    Parameters
    ----------
    data : DataFrame
        Input data containing ``x``, ``y``, and (optionally) ``C``.
    x, y : str
        Column names for the bivariate axes.
    C : str, optional
        Column reduced per hex instead of counting. When ``None`` (the
        default), the color encodes the per-hex count.
    reduce_C_function : callable, default :func:`numpy.mean`
        Aggregator applied to the values of ``C`` within each hex.
        Ignored when ``C`` is None.
    gridsize : int or (int, int), default 30
        Number of hexagons along x (and y, if a tuple is passed). The
        matplotlib default of 100 is too fine for publiplots' mm-sized
        axes; 30 is legible at the 70×50 mm baseline.
    bins : {None, 'log', int, sequence}, optional
        Passed through to :meth:`matplotlib.axes.Axes.hexbin`. The
        special string ``'log'`` log-normalizes the color scale, which is
        the usual choice for heavy-tailed densities.
    mincnt : int, default 1
        Hide hexes below this count (matplotlib returns a masked array,
        so empty hexes render as fully transparent cells — matching
        seaborn's appearance). Pass ``0`` to render every hex.
    cmap : str or Colormap, optional
        Colormap for the hex density. When ``None`` (the default),
        builds a light sequential gradient from ``pp.rcParams["color"]``
        so the default look matches the rest of publiplots' theme.
        Pass any matplotlib/seaborn cmap name (``"viridis"``,
        ``"magma"``, ``"rocket"``...) to override.
    vmin, vmax : float, optional
        Color scale bounds. When both are ``None`` (the default),
        matplotlib autoscales from the reduced/count array.
    edgecolor : str, optional
        Edge color for each hex cell. Falls back to
        ``pp.rcParams["edgecolor"]``; when that is also ``None``, edges
        are not drawn (hexbin's default — stroking every cell rarely
        reads well at publication sizes).
    linewidth : float, optional
        Edge width. Falls back to ``pp.rcParams["lines.linewidth"]``.
    alpha : float, default 1.0
        Face transparency for the hex cells. Unlike marker-based plots,
        hexbin cells are solid density patches — the publiplots
        ``rcParams["alpha"]`` default (tuned for layered bars / scatter)
        would wash them out, so this kwarg defaults to 1.0 instead.
    extent : (xmin, xmax, ymin, ymax), optional
        Data-coordinate rectangle used for binning. Defaults to the data
        range.
    ax : Axes, optional
        Target axes. When ``None``, a new figure is created via
        :func:`publiplots.subplots`.
    title : str, default ""
        Plot title.
    xlabel, ylabel : str, default ""
        Axis labels. ``None`` preserves whatever matplotlib set.
    legend : bool or dict, default True
        ``True`` stashes and renders a colorbar for the hue dimension.
        ``False`` stashes nothing. A dict maps legend kinds to booleans
        (hexbin only emits the ``"hue"`` kind).
    legend_kws : dict, optional
        Forwarded to the legend builder (e.g.
        ``{'inside': True, 'loc': 'upper right'}`` for an in-axes colorbar,
        or ``{'hue_label': 'log N'}`` to override the legend title).
    **kwargs
        Extra keyword arguments forwarded to
        :meth:`matplotlib.axes.Axes.hexbin`. ``figsize`` is rejected.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Count-density hexbin (the common case for dense scatter):

    >>> ax = pp.hexbinplot(data=df, x="umap1", y="umap2")

    Color each hex by the mean of a third column:

    >>> ax = pp.hexbinplot(data=df, x="umap1", y="umap2",
    ...                    C="score", reduce_C_function=np.mean)

    Log-scaled density on a heavy-tailed distribution:

    >>> ax = pp.hexbinplot(data=df, x="x", y="y", bins="log")

    See Also
    --------
    publiplots.scatterplot : Use when points are sparse enough to read
        individually.
    publiplots.heatmap : 2D matrix visualization on a pre-aggregated grid.
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    linewidth = resolve_param("lines.linewidth", linewidth)
    edgecolor = resolve_param("edgecolor", edgecolor)
    cmap = resolve_continuous_cmap(cmap)
    hex_edgecolor = edgecolor if edgecolor is not None else "none"

    required_cols = [x, y] + ([C] if C is not None else [])
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        _, ax = _pp_subplots()

    x_arr = np.asarray(data[x].values, dtype=float)
    y_arr = np.asarray(data[y].values, dtype=float)
    C_arr = np.asarray(data[C].values, dtype=float) if C is not None else None

    hexbin_kwargs = dict(
        gridsize=gridsize,
        bins=bins,
        mincnt=mincnt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=hex_edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    if extent is not None:
        hexbin_kwargs["extent"] = extent
    if C_arr is not None:
        hexbin_kwargs["C"] = C_arr
        hexbin_kwargs["reduce_C_function"] = reduce_C_function
    hexbin_kwargs.update(kwargs)

    collection = ax.hexbin(x_arr, y_arr, **hexbin_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    _legend(
        ax=ax,
        collection=collection,
        C=C,
        legend=legend,
        legend_kws=legend_kws,
    )

    return ax


def _legend(
    ax: Axes,
    collection,
    C: Optional[str],
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
) -> None:
    """Stash a continuous-hue entry for the hexbin colorbar and render.

    Reuses the collection's own cmap + norm so autoscale (``vmin``/``vmax``
    left as None) and ``bins='log'`` are honored without re-deriving the
    normalization.
    """
    if legend is False:
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})
    hue_label = legend_kws.pop("hue_label", C if C is not None else "count")

    if flags["hue"]:
        stash_continuous_hue(
            ax,
            name=hue_label,
            palette=collection.get_cmap(),
            hue_norm=collection.norm,
        )

    render_entries(ax, flags=flags, legend_kws=legend_kws)
