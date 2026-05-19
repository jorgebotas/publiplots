"""
Box plot functions for publiplots.

This module provides publication-ready box plot visualizations with
transparent fill and opaque edges.
"""

import warnings
from typing import Optional, List, Dict, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
import seaborn as sns
import pandas as pd
import numpy as np

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils.max_width import clamp_patch_widths_mm
from publiplots.utils.rounding import apply_border_radius, normalize_border_radius
from publiplots.utils.transparency import ArtistTracker
from publiplots.utils import is_categorical
from publiplots.utils.plot_legend import stash_hue_legend


def boxplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    orient: Optional[str] = None,
    color: Optional[str] = None,
    linecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    width: float = 0.8,
    gap: float = 0,
    whis: float = 1.5,
    showcaps: bool = False,
    fliersize: Optional[float] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    border_radius: Optional[Union[float, Tuple[float, float]]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    annotate: Union[bool, Dict, None] = None,
    **kwargs
) -> Axes:
    """
    Create a publication-ready box plot.

    This function creates box plots with transparent fill and opaque edges,
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
        Orientation of the plot ('v' or 'h').
    color : str, optional
        Fixed color for all boxes (only used when hue is None).
    linecolor : str, optional
        Deprecated. Use edgecolor instead. Color of the box edges.
    edgecolor : str, optional
        Color of the box edges, whiskers, caps, and outlier marker edges.
        When None, edges match the palette face color for each group.
    palette : str, dict, or list, optional
        Color palette for hue grouping.
    width : float, default=0.8
        Width of the boxes.
    gap : float, default=0
        Gap between boxes when using hue.
    whis : float, default=1.5
        Proportion of IQR past low and high quartiles to extend whiskers.
    showcaps: bool, default=False
        Whether to show the caps.
    fliersize : float, optional
        Size of outlier markers.
    linewidth : float, optional
        Width of box edges. When None, resolved from
        ``publiplots.rcParams["lines.linewidth"]``.
    alpha : float, optional
        Transparency of box fill (0-1). When None, resolved from
        ``publiplots.rcParams["alpha"]``.
    border_radius : float or (top_mm, bottom_mm) tuple, optional
        Corner radius for the IQR box, in **millimeters** (print-consistent,
        independent of the data-axis range). A scalar rounds all four corners
        symmetrically; a 2-tuple rounds top and bottom independently
        (e.g. ``border_radius=(1.5, 0)`` keeps the Q1 edge square — useful
        when the box is visually paired with a density cloud in
        ``pp.raincloudplot``). Defaults to the ``box.border_radius`` rcParam
        (``(0, 0)`` = flat). Set globally via
        ``pp.rcParams['box.border_radius'] = 1.5``.
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
    **kwargs
        Additional keyword arguments passed to seaborn.boxplot.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple box plot:

    >>> import publiplots as pp
    >>> ax = pp.boxplot(data=df, x="category", y="value")

    Box plot with hue grouping:

    >>> ax = pp.boxplot(
    ...     data=df, x="category", y="value", hue="group"
    ... )

    Box plot with statistical annotations between groups:

    >>> ax = pp.boxplot(
    ...     data=df, x="category", y="value",
    ...     annotate={"comparisons": [("A", "B"), ("B", "C")]}
    ... )

    See Also
    --------
    publiplots.violinplot : Distribution-shape alternative with the same interface.
    publiplots.raincloudplot : Box + half-violin + strip composite.
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
    if edgecolor is not None and linecolor is not None:
        warnings.warn(
            "linecolor is deprecated in favor of edgecolor. "
            "edgecolor takes precedence when both are provided.",
            FutureWarning,
            stacklevel=2,
        )
    resolved_edgecolor = edgecolor if edgecolor is not None else linecolor

    # 1D mode: synthesize a constant categorical column on the missing axis so
    # the rest of the function flows through the 2D code path unchanged.
    if x is None and y is None:
        raise ValueError("at least one of `x` or `y` must be provided to pp.boxplot.")
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

    # Determine categorical axis
    categorical_axis = "x"  # default
    if x is not None and y is not None:
        categorical_axis = "x" if is_categorical(data[x]) else "y"
    elif orient == "h":
        categorical_axis = "y"

    # Prepare kwargs for seaborn boxplot
    boxplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "orient": orient,
        "color": color if hue is None else None,
        "linecolor": resolved_edgecolor,
        "palette": palette if hue else None,
        "width": width,
        "gap": gap,
        "whis": whis,
        "showcaps": showcaps,
        "fliersize": fliersize,
        "linewidth": linewidth,
        "fill": True,  # Need fill=True to get patches
        "ax": ax,
        "legend": False,  # Handle legend ourselves
    }

    # Merge with user-provided kwargs
    boxplot_kwargs.update(kwargs)

    # Track artists before plotting
    tracker = ArtistTracker(ax)

    # Create boxplot
    sns.boxplot(**boxplot_kwargs)

    # Get newly created patches and lines
    new_patches = tracker.get_new_patches()
    new_lines = tracker.get_new_lines()

    # Build a map of position -> color from patches
    # Position is on the categorical axis (x or y)
    patch_colors = {}
    for patch in new_patches:
        verts = patch.get_path().vertices
        if categorical_axis == "x":
            pos = round(np.mean(verts[:, 0]), 2)
        else:
            pos = round(np.mean(verts[:, 1]), 2)
        patch_colors[pos] = patch.get_facecolor()

    # Resolve markeredgewidth for outliers
    flierprops = kwargs.get("flierprops", {})
    markeredgewidth = flierprops.get("markeredgewidth", None)
    markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)

    # Recolor lines based on position and edgecolor
    for line in new_lines:
        line_data = line.get_xdata() if categorical_axis == "x" else line.get_ydata()
        if len(line_data) == 0:
            continue
        pos = np.mean(line_data)
        closest_pos = min(patch_colors.keys(), key=lambda p: abs(p - pos))
        base_color = patch_colors[closest_pos]

        if line.get_marker() and line.get_marker() != 'None':
            # Outlier markers: face = palette color, edge = edgecolor or palette
            line.set_markerfacecolor(base_color)
            line.set_markeredgecolor(resolved_edgecolor if resolved_edgecolor else base_color)
            line.set_markeredgewidth(markeredgewidth)
        else:
            # Structural lines (whiskers, caps, medians)
            line.set_color(resolved_edgecolor if resolved_edgecolor else base_color)
            line.set_linewidth(linewidth)

    # Set edge colors on box patches
    for patch in new_patches:
        patch.set_edgecolor(resolved_edgecolor if resolved_edgecolor else patch.get_facecolor())

    # Cap box width (mm) BEFORE apply_border_radius so the rounding swap
    # reads the clamped extents. categorical_axis is "x" (vertical boxes)
    # or "y" (horizontal); pass directly to the helper.
    _max_box_mm = resolve_param("box.max_width", None)
    if _max_box_mm is not None and _max_box_mm > 0:
        # Snapshot pre-clamp patch extents so we can scale the
        # whisker / cap / median lines toward the (now-clamped) patch
        # center on the categorical axis.
        _pre_extents = []
        for p in tracker.get_new_patches():
            bbox = p.get_path().get_extents()
            _pre_extents.append(bbox)
        clamp_patch_widths_mm(
            tracker.get_new_patches(),
            _max_box_mm,
            ax,
            axis=categorical_axis,
        )
        _post_extents = [p.get_path().get_extents() for p in tracker.get_new_patches()]
        # For each line that lives within an old patch's categorical-axis
        # extent, scale the line toward the patch's center to match the
        # new (clamped) extent. Whiskers (single x-point) are unaffected.
        for line in tracker.get_new_lines():
            if categorical_axis == "x":
                xs = np.asarray(line.get_xdata(), dtype=float)
            else:
                xs = np.asarray(line.get_ydata(), dtype=float)
            if len(xs) == 0:
                continue
            cmin, cmax = float(xs.min()), float(xs.max())
            extent = cmax - cmin
            for pre, post in zip(_pre_extents, _post_extents):
                if categorical_axis == "x":
                    pre_min, pre_max = pre.x0, pre.x1
                    post_min, post_max = post.x0, post.x1
                else:
                    pre_min, pre_max = pre.y0, pre.y1
                    post_min, post_max = post.y0, post.y1
                pre_w = pre_max - pre_min
                if pre_w <= 0:
                    continue
                # Line "belongs" to this patch if its center is inside the
                # pre-clamp extent and its full span is within (slightly
                # tolerant) the patch width.
                lc = (cmin + cmax) / 2.0
                if not (pre_min - 1e-9 <= lc <= pre_max + 1e-9):
                    continue
                if extent > pre_w + 1e-9:
                    continue
                # Compute scale by patch shrinkage; scale line toward patch center.
                post_w = post_max - post_min
                if post_w >= pre_w - 1e-12:
                    break  # patch unchanged → no clamp needed
                center = (post_min + post_max) / 2.0
                scale = post_w / pre_w
                new_xs = center + (xs - (pre_min + pre_max) / 2.0) * scale
                if categorical_axis == "x":
                    line.set_xdata(new_xs)
                else:
                    line.set_ydata(new_xs)
                break

    # Round the IQR-box corners per rcParam / kwarg. No-op when (0, 0).
    # Runs AFTER the edgecolor loop so face/edge are copied onto the new
    # patch, and BEFORE apply_transparency so the tracker snapshot diff
    # picks up the swapped-in _RoundedBarPatch(es). Mirrors bar.py.
    # Non-PathPatch artists (whisker/cap Line2Ds) are not in ax.patches
    # and are silently ignored by apply_border_radius.
    radius = normalize_border_radius(
        resolve_param("box.border_radius", border_radius)
    )
    apply_border_radius(tracker.get_new_patches(), radius, ax)

    # Apply transparency to box patch faces
    tracker.apply_transparency(on="patches", face_alpha=alpha)
    # Apply transparency to outlier marker faces only
    for line in new_lines:
        if line.get_marker() and line.get_marker() != 'None':
            fc = line.get_markerfacecolor()
            line.set_markerfacecolor(to_rgba(fc, alpha))

    # Stash legend entries and optionally render per-axis legend.
    # legend may be bool or dict[str, bool]; False short-circuits both.
    _stash_legend(
        ax=ax,
        hue=hue,
        palette=palette,
        edgecolor=resolved_edgecolor,
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
    from publiplots.annotate._builders import build_from_boxplot_call
    if is_categorical(data[x]):
        categorical_axis = x
    elif is_categorical(data[y]):
        categorical_axis = y
    else:
        categorical_axis = x
    ax._publiplots_box_meta = build_from_boxplot_call(
        ax=ax, data=data, x=x, y=y, hue=hue,
        categorical_axis=categorical_axis,
        palette=palette if isinstance(palette, dict) else None,
        whis=whis,
        source_frame=_source_data,
    )
    if annotate:
        from publiplots.annotate import annotate as _annotate_fn
        opts = dict(annotate) if isinstance(annotate, dict) else {}
        kind = opts.pop("kind", "box_stats")
        _annotate_fn(ax, kind=kind, **opts)

    if _hide_axis == "x":
        ax.set_xticks([])
    elif _hide_axis == "y":
        ax.set_yticks([])

    return ax


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
