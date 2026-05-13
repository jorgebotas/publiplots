"""
Violin plot functions for publiplots.

This module provides publication-ready violin plot visualizations with
transparent fill and opaque edges.
"""

from typing import Optional, List, Dict, Tuple, Union

from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.collections import LineCollection

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import seaborn as sns
import pandas as pd
import numpy as np

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils import is_categorical
from publiplots.utils.plot_legend import stash_hue_legend
from publiplots.utils.transparency import ArtistTracker


def violinplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    orient: Optional[str] = None,
    color: Optional[str] = None,
    edgecolor: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    saturation: float = 1.0,
    fill: bool = False,
    inner: Optional[str] = "box",
    split: bool = False,
    width: float = 0.8,
    dodge: Union[bool, str] = "auto",
    gap: float = 0,
    linewidth: Optional[float] = None,
    linecolor: str = "auto",
    cut: float = 2,
    gridsize: int = 100,
    bw_method: str = "scott",
    bw_adjust: float = 1,
    density_norm: str = "area",
    common_norm: bool = False,
    alpha: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    annotate: Union[bool, Dict, None] = None,
    side: str = "both",
    **kwargs
) -> Axes:
    """
    Create a publication-ready violin plot.

    This function creates violin plots with transparent fill and opaque edges,
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
    orient : str, optional
        Deprecated. Orientation is inferred from which of ``x`` / ``y`` is
        categorical; passing a non-None value raises ``DeprecationWarning``.
    color : str, optional
        Fixed color for all violins (only used when hue is None).
    edgecolor : str, optional
        Color of violin edges. When provided, overrides linecolor.
        Preferred over linecolor (which is deprecated in favor of edgecolor).
    palette : str, dict, or list, optional
        Color palette for hue grouping.
    saturation : float, default=1.0
        Proportion of the original saturation to draw colors at.
    fill : bool, default=False
        Whether to fill the violin interior.
    inner : str, optional, default="box"
        Representation of the data in the violin interior.
        Options: "box", "quart", "point", "stick", None.
    split : bool, default=False
        When using hue nesting with a variable that takes two levels,
        setting split to True will draw half of a violin for each level.
    width : float, default=0.8
        Width of the violins.
    dodge : bool or "auto", default="auto"
        When hue nesting is used, whether elements should be shifted along
        the categorical axis.
    gap : float, default=0
        Gap between violins when using hue.
    linewidth : float, optional
        Width of violin edges. When None, resolved from
        ``publiplots.rcParams["lines.linewidth"]``.
    linecolor : str, default="auto"
        Deprecated. Use ``edgecolor`` instead. Kept for backward
        compatibility; when ``edgecolor`` is also set, ``edgecolor`` wins.
    cut : float, default=2
        Distance past extreme data points to extend density estimate.
    gridsize : int, default=100
        Number of points in the discrete grid used to evaluate KDE.
    bw_method : str, default="scott"
        Method for calculating smoothing bandwidth.
    bw_adjust : float, default=1
        Factor to adjust the bandwidth.
    density_norm : str, default="area"
        Method for normalizing density ("area", "count", "width").
    common_norm : bool, default=False
        When True, normalize across the entire dataset.
    alpha : float, optional
        Transparency of violin fill (0-1). When None, resolved from
        ``publiplots.rcParams["alpha"]``.
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label.
    ylabel : str, default=""
        Y-axis label.
    legend : bool or dict, default=True
        Whether to show the legend. Accepts ``bool`` or ``dict[kind, bool]``
        for per-kind control (e.g., ``legend={"hue": False}``).
    legend_kws : dict, optional
        Additional keyword arguments for legend.
    annotate : bool or dict, optional
        If truthy, run :func:`publiplots.annotate` with ``kind="box_stats"``
        on the resulting axes. A dict is forwarded as keyword arguments to
        the annotate call (e.g., ``annotate={"comparisons": [("A", "B")]}``).
    side : str, default="both"
        Publiplots-specific: which side of the categorical position to draw
        the violin on. Options: ``"both"``, ``"left"``, ``"right"``. When
        ``"left"`` or ``"right"``, draws only half of the violin — this is
        what :func:`publiplots.raincloudplot` uses to build its "cloud".
        Note: when ``side`` is not ``"both"`` and ``hue`` is specified, the
        hue coloring is applied but split behavior is controlled by ``side``.
    **kwargs
        Additional keyword arguments passed to seaborn.violinplot.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple violin plot:

    >>> import publiplots as pp
    >>> ax = pp.violinplot(data=df, x="category", y="value")

    Violin plot with hue grouping:

    >>> ax = pp.violinplot(
    ...     data=df, x="category", y="value", hue="group"
    ... )

    Half-violin (used internally by raincloudplot):

    >>> ax = pp.violinplot(data=df, x="category", y="value", side="right")

    See Also
    --------
    publiplots.boxplot : Box-and-whisker alternative with the same interface.
    publiplots.raincloudplot : Composite of half-violin + box + strip.
    publiplots.annotate : Statistical annotations (used via ``annotate=``).
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Resolve edgecolor vs linecolor (backward compat)
    if edgecolor is not None and linecolor != "auto":
        import warnings
        warnings.warn(
            "linecolor is deprecated in favor of edgecolor. "
            "edgecolor takes precedence when both are provided.",
            FutureWarning,
            stacklevel=2,
        )
    if edgecolor is not None:
        linecolor = edgecolor

    # 1D mode: synthesize a constant categorical column on the missing axis so
    # the rest of the function flows through the 2D code path unchanged.
    if x is None and y is None:
        raise ValueError("at least one of `x` or `y` must be provided to pp.violinplot.")
    _hide_axis: Optional[str] = None
    if x is None or y is None:
        _DUMMY = "__publiplots_constant_axis__"
        data = data.assign(**{_DUMMY: ""})
        if x is None:
            x = _DUMMY
            _hide_axis = "x"
        else:
            y = _DUMMY
            _hide_axis = "y"

    # Validate side parameter
    if side not in ("both", "left", "right"):
        raise ValueError(f"side must be 'both', 'left', or 'right', got '{side}'")

    if orient is not None:
        raise DeprecationWarning("orient is deprecated. Use x and y instead.")

    # Preserve the caller's original DataFrame identity for downstream
    # annotate builders that stash `source_frame` on the meta.
    _source_data = data

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

    # Resolve linewidth for box whiskers
    inner_kws = kwargs.pop("inner_kws", {})
    if inner == "box":
        inner_kws["whis_width"] = inner_kws.get("whis_width", linewidth)

    # Prepare kwargs for seaborn violinplot
    violinplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        # "orient": orient,
        "color": color if hue is None else None,
        "palette": palette if hue else None,
        "saturation": saturation,
        "fill": fill,
        "inner": inner,
        "split": split,
        "width": width,
        "dodge": dodge,
        "gap": gap,
        "linewidth": linewidth,
        "linecolor": linecolor,
        "cut": cut,
        "gridsize": gridsize,
        "bw_method": bw_method,
        "bw_adjust": bw_adjust,
        "density_norm": density_norm,
        "common_norm": common_norm,
        "inner_kws": inner_kws,
        "ax": ax,
        "legend": False,  # Handle legend ourselves
    }

    # Merge with user-provided kwargs
    violinplot_kwargs.update(kwargs)

    # Track artists before plotting
    tracker = ArtistTracker(ax)

    # Create violinplot
    sns.violinplot(**violinplot_kwargs)

    # Side clip the violin
    if side != "both":
        # Determine orientation for side clipping
        is_vertical = is_categorical(data[x])
        _side_clip_violin(tracker, side, is_vertical, inner)

    # Apply transparency only to new violin collections
    tracker.apply_transparency(on="collections", face_alpha=alpha)

    # Override edge color on the violin PolyCollections. Seaborn's linecolor
    # only strokes the inner stat lines; the FillBetweenPolyCollection edges
    # otherwise keep the palette color and ignore our edgecolor request.
    # This MUST run after apply_transparency — otherwise the transparency
    # helper sees empty face_colors (fill=False) and falls back to our
    # edgecolor, turning faces into alpha-dimmed versions of the edge.
    if edgecolor is not None:
        for coll in tracker.get_new_collections():
            if isinstance(coll, FillBetweenPolyCollection):
                coll.set_edgecolors(edgecolor)

    # Stash legend entries and optionally render per-axis legend.
    # legend may be bool or dict[str, bool]; False short-circuits both.
    _stash_legend(
        ax=ax,
        hue=hue,
        palette=palette,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        legend=legend,
        legend_kws=legend_kws,
    )

    # Set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # Always attach the cache so follow-up pp.annotate(ax, ...) calls work.
    from publiplots.annotate._builders import build_from_violinplot_call
    if is_categorical(data[x]):
        categorical_axis = x
    elif is_categorical(data[y]):
        categorical_axis = y
    else:
        categorical_axis = x
    ax._publiplots_box_meta = build_from_violinplot_call(
        ax=ax, data=data, x=x, y=y, hue=hue,
        categorical_axis=categorical_axis,
        palette=palette if isinstance(palette, dict) else None,
        source_frame=_source_data,
    )
    if annotate:
        from publiplots.annotate import annotate as _annotate_fn
        opts = dict(annotate) if isinstance(annotate, dict) else {}
        kind = opts.pop("kind", "box_stats")
        _annotate_fn(ax, kind=kind, **opts)

    if _hide_axis == "x":
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    elif _hide_axis == "y":
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    return ax


def _side_clip_violin(
    tracker: ArtistTracker,
    side: str,
    is_vertical: bool,
    inner: str,
) -> None:
    """
    Side clip a violin plot.

    Parameters
    ----------
    tracker : ArtistTracker
        Artist tracker object.
    side : str
        Side to clip the violin on.
    is_vertical : bool
        Whether the violin is vertical.
    inner : str
        Inner type of the violin.
    """
    new_collections = tracker.get_new_collections()
    for coll in new_collections:
            if not isinstance(coll, FillBetweenPolyCollection):
                continue
            paths = coll.get_paths()
            new_paths = []
            for path in paths:
                vertices = path.vertices
                codes = path.codes if path.codes is not None else None

                if is_vertical:
                    # Find center x position (categorical axis)
                    center_x = np.mean(vertices[:, 0])

                    if side == "right":
                        # Keep only right half (x >= center)
                        mask = vertices[:, 0] >= center_x
                        # Clip vertices to center
                        new_vertices = vertices.copy()
                        new_vertices[:, 0] = np.maximum(vertices[:, 0], center_x)
                    else:  # left
                        # Keep only left half (x <= center)
                        mask = vertices[:, 0] <= center_x
                        new_vertices = vertices.copy()
                        new_vertices[:, 0] = np.minimum(vertices[:, 0], center_x)
                else:
                    # Horizontal orientation - clip on y axis
                    center_y = np.mean(vertices[:, 1])

                    if side == "right":
                        # Keep only top half (y >= center)
                        new_vertices = vertices.copy()
                        new_vertices[:, 1] = np.maximum(vertices[:, 1], center_y)
                    else:  # left
                        # Keep only bottom half (y <= center)
                        new_vertices = vertices.copy()
                        new_vertices[:, 1] = np.minimum(vertices[:, 1], center_y)

                new_paths.append(new_vertices)

            # set_paths expects list of vertices arrays for PolyCollection
            coll.set_verts(new_paths)

    # Clip inner lines (quart, quartile) to match half-violin
    if inner in ("quart", "quartile"):
        new_lines = tracker.get_new_lines()
        for line in new_lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()

            if len(xdata) == 0:
                continue

            if is_vertical:
                # For vertical violins, clip the x-coordinates of horizontal lines
                center_x = np.mean(xdata)
                if side == "right":
                    new_xdata = np.maximum(xdata, center_x)
                else:  # left
                    new_xdata = np.minimum(xdata, center_x)
                line.set_xdata(new_xdata)
            else:
                # For horizontal violins, clip the y-coordinates
                center_y = np.mean(ydata)
                if side == "right":
                    new_ydata = np.maximum(ydata, center_y)
                else:  # left
                    new_ydata = np.minimum(ydata, center_y)
                line.set_ydata(new_ydata)

    # Clip inner sticks (LineCollection) to match half-violin
    if inner in ("stick", "sticks"):
        for coll in new_collections:
            if not isinstance(coll, LineCollection):
                continue

            segments = coll.get_segments()
            new_segments = []

            for segment in segments:
                # Each segment is an array of points [[x1, y1], [x2, y2], ...]
                new_segment = segment.copy()

                if is_vertical:
                    center_x = np.mean(segment[:, 0])
                    if side == "right":
                        new_segment[:, 0] = np.maximum(segment[:, 0], center_x)
                    else:  # left
                        new_segment[:, 0] = np.minimum(segment[:, 0], center_x)
                else:
                    center_y = np.mean(segment[:, 1])
                    if side == "right":
                        new_segment[:, 1] = np.maximum(segment[:, 1], center_y)
                    else:  # left
                        new_segment[:, 1] = np.minimum(segment[:, 1], center_y)

                new_segments.append(new_segment)

            coll.set_segments(new_segments)


def _stash_legend(
        ax: Axes,
        hue: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        edgecolor: Optional[str],
        alpha: Optional[float],
        linewidth: Optional[float],
        legend: Union[bool, Dict],
        legend_kws: Optional[Dict],
    ) -> None:
    """Delegate to the shared hue-legend helper."""
    stash_hue_legend(
        ax,
        hue=hue,
        palette=palette,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        legend=legend,
        legend_kws=legend_kws,
    )
