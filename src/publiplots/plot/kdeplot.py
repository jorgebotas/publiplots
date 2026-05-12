"""
Kernel density estimate (KDE) plot functions for publiplots.

This module provides publication-ready kernel density estimates in both
univariate (1D curve / filled density) and bivariate (2D contour) modes,
with optional hue grouping, continuous-hue colorbar legend, and the
publiplots double-layer transparent-fill styling shared with
:func:`publiplots.histplot`.

Dispatch rule
-------------
- Univariate (1D): exactly one of ``x``/``y`` is provided.
- Bivariate (2D): both ``x`` and ``y`` are provided (contour plot).

Legend stashing
---------------
- 1D + categorical hue: one ``LegendEntry`` with rectangle handles when
  ``fill=True``, line handles when ``fill=False``.
- 2D + categorical hue: one ``LegendEntry`` with line handles (one per
  hue level, colored from the palette).
- 2D + ``cbar=True`` (no hue): one continuous-hue ``LegendEntry`` built
  from the QuadContourSet's cmap + data-level-derived normalization.
  Seaborn's own colorbar is suppressed; the band is rendered by the
  publiplots legend reactor.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D

from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.rcparams import resolve_param
from publiplots.utils import create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    resolve_legend_flags,
    stash_entry,
)
from publiplots.utils.plot_legend import render_entries, stash_continuous_hue
from publiplots.utils.transparency import ArtistTracker


def kdeplot(
    data: Optional[pd.DataFrame] = None,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    fill: Optional[bool] = None,
    multiple: str = "layer",
    common_norm: bool = True,
    common_grid: bool = False,
    cumulative: bool = False,
    weights: Optional[str] = None,
    bw_method: Union[str, float] = "scott",
    bw_adjust: float = 1,
    gridsize: int = 200,
    cut: float = 3,
    clip: Optional[Tuple[float, float]] = None,
    log_scale: Optional[Union[bool, float, Tuple]] = None,
    warn_singular: bool = True,
    levels: Union[int, List[float]] = 10,
    thresh: float = 0.05,
    cbar: bool = False,
    cbar_ax: Optional[Axes] = None,
    cbar_kws: Optional[Dict] = None,
    cmap: Optional[str] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs,
) -> Axes:
    """
    Create a publication-ready kernel density estimate (KDE) plot.

    Draws a univariate density curve when exactly one of ``x`` / ``y``
    is provided, or a bivariate contour density when both are. Supports
    hue grouping, the full seaborn ``kdeplot`` estimator API (bandwidth,
    cut, clip, cumulative, log-scale), and publiplots' double-layer
    transparent-fill styling + standard legend reactor.

    Parameters
    ----------
    data : DataFrame, optional
        Long-form dataframe with ``x`` / ``y`` / ``hue`` columns.
    x : str, optional
        Column for the x-axis variable. Provide alone for a 1D density
        curve; provide with ``y`` for a 2D contour plot.
    y : str, optional
        Column for the y-axis variable. Provide alone for a horizontal
        1D density curve; provide with ``x`` for a 2D contour plot.
    hue : str, optional
        Column name for categorical color grouping.
    palette : str, dict, or list, optional
        Color palette. Accepts a palette name (publiplots or seaborn),
        an explicit list of colors, or a ``{hue_level: color}`` dict.
    color : str, optional
        Fixed color used when ``hue`` is None.
    alpha : float, optional
        Transparency for filled regions (0-1). Falls back to
        ``publiplots.rcParams["alpha"]``.
    fill : bool, optional
        If True, fill area under curves (1D) or color-fill contour bands
        (2D). Pass ``None`` to use seaborn's derived default (which is
        ``False`` for ``multiple='layer'`` and ``True`` for
        ``multiple='stack'`` / ``'fill'``).
    multiple : {'layer', 'stack', 'fill'}, default='layer'
        Handling of overlapping hue levels for 1D densities.
    common_norm : bool, default=True
        When normalizing, normalize across all hue levels jointly (True)
        or per-level (False).
    common_grid : bool, default=False
        Use a shared evaluation grid across hue levels (useful when
        ``multiple='stack'`` or ``'fill'``).
    cumulative : bool, default=False
        Plot the cumulative distribution instead of the density.
    weights : str, optional
        Column name with per-observation weights.
    bw_method : str or float, default='scott'
        Bandwidth rule (``'scott'`` / ``'silverman'``) or an explicit
        scalar.
    bw_adjust : float, default=1
        Multiplicative factor on the automatic bandwidth.
    gridsize : int, default=200
        Number of evaluation points for the KDE grid.
    cut : float, default=3
        Extend KDE beyond data range by ``cut * bandwidth``.
    clip : (min, max), optional
        Hard limits for the KDE evaluation range.
    log_scale : bool, number, or pair, optional
        Log-scale the value axis; forwarded to seaborn.
    warn_singular : bool, default=True
        Emit a warning when a hue level has near-zero variance (KDE
        cannot be computed).
    levels : int or list of float, default=10
        Number of contour levels (int) or explicit probability-mass
        cutoffs (list). Used in 2D mode only.
    thresh : float, default=0.05
        Lowest iso-density level as a fraction of peak density. Used in
        2D mode only.
    cbar : bool, default=False
        2D only. If True, draws filled contour bands colored by density
        and stashes a continuous-hue colorbar entry on the axes. The
        band is rendered by publiplots' legend reactor, not by seaborn.
        When ``hue`` is None and ``fill`` is unset, ``cbar=True``
        implicitly switches to ``fill=True`` so the contour fill encodes
        the density value the colorbar advertises.
    cbar_ax : Axes, optional
        Accepted for signature compatibility. Logs a UserWarning — the
        colorbar is placed via ``stash_continuous_hue``, not onto
        ``cbar_ax`` directly.
    cbar_kws : dict, optional
        Forwarded into ``legend_kws`` (e.g. ``{'hue_label': 'density'}``).
    cmap : str or Colormap, optional
        2D only. Colormap used to color the filled contour bands and
        the colorbar gradient. When None (the default), falls back to
        ``matplotlib.rcParams["image.cmap"]`` — same convention as
        :func:`pp.hexbinplot`. Only applies in 2D mode without ``hue``.
    linewidth : float, optional
        Stroke width for curves / contour lines. Falls back to rcParams.
    edgecolor : str, optional
        Override for curve / contour edge color. Falls back to rcParams.
    ax : Axes, optional
        Target axes. When None, a new figure is created via
        :func:`publiplots.subplots`.
    title : str, optional
        Plot title. When None, no title is set.
    xlabel : str, optional
        X-axis label. When None (the default), seaborn's inferred label
        (the column name or ``'Density'``) is preserved.
    ylabel : str, optional
        Y-axis label. Same semantics as ``xlabel``.
    legend : bool or dict, default=True
        Legend control. ``True`` stashes and renders all legend kinds.
        ``False`` hides everything. A dict enables per-kind
        (e.g. ``{"hue": False}``).
    legend_kws : dict, optional
        Extra kwargs forwarded to the legend builder
        (e.g. ``{'inside': True}``).
    **kwargs
        Extra kwargs passed to :func:`seaborn.kdeplot`. ``figsize`` is
        rejected — use ``pp.subplots(axes_size=...)`` and pass ``ax=``.

    Returns
    -------
    Axes
        The axes the plot was drawn on.

    Examples
    --------
    >>> ax = pp.kdeplot(data=df, x="value")
    >>> ax = pp.kdeplot(data=df, x="value", hue="group", fill=True)
    >>> ax = pp.kdeplot(data=df, x="x", y="y", cbar=True)

    See Also
    --------
    publiplots.histplot : Binned counterpart with optional KDE overlay.
    publiplots.hexbinplot : Bivariate hex binning for dense point clouds.
    seaborn.kdeplot : Underlying rendering primitive.
    """
    from publiplots.layout.subplots import reject_figsize, subplots as _pp_subplots
    reject_figsize(kwargs)

    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Dispatch: exactly one of (x alone), (y alone), or (x and y) is valid.
    if x is None and y is None:
        raise ValueError(
            "pp.kdeplot requires at least one of `x` or `y` to be provided."
        )
    is_2d = x is not None and y is not None

    # Column-presence validation (mimics sns/pp.hexbinplot semantics).
    if data is not None:
        required_cols: List[str] = []
        for name in (x, y, hue, weights):
            if name is not None:
                required_cols.append(name)
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

    if ax is None:
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Signature-compat: cbar_ax is accepted but not honored; we route
    # colorbars through stash_continuous_hue, not manual placement.
    if cbar_ax is not None:
        warnings.warn(
            "pp.kdeplot: cbar_ax= is ignored; colorbar placement is "
            "managed by publiplots' legend reactor. Use pp.legend(side=...) "
            "or legend_kws={'inside': True} to customize placement.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve categorical palette map once up-front so seaborn + legend
    # agree on per-level colors.
    palette_map: Optional[Dict] = None
    if hue is not None and data is not None:
        palette_map = resolve_palette_map(
            values=list(data[hue].unique()),
            palette=palette,
        )

    tracker = ArtistTracker(ax)

    sns_kwargs: Dict = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "weights": weights,
        "multiple": multiple,
        "common_norm": common_norm,
        "common_grid": common_grid,
        "cumulative": cumulative,
        "bw_method": bw_method,
        "bw_adjust": bw_adjust,
        "gridsize": gridsize,
        "cut": cut,
        "clip": clip,
        "log_scale": log_scale,
        "warn_singular": warn_singular,
        "levels": levels,
        "thresh": thresh,
        # IMPORTANT: always suppress seaborn's colorbar; publiplots manages it.
        "cbar": False,
        "ax": ax,
        "legend": False,
    }
    # seaborn rejects linewidth in its 2D contour path (it forwards to
    # matplotlib's contour, which doesn't accept it), but honors it in 1D.
    if not is_2d:
        sns_kwargs["linewidth"] = linewidth

    # 2D + cbar + no hue: a colorbar is meaningful only when the
    # contours are actually colored by density. Default fill=True (when
    # the user didn't explicitly set fill) and resolve a continuous
    # cmap so the bands span the density range. Without this, seaborn
    # draws all contour lines in a single color and the colorbar
    # advertises a gradient that the plot does not actually encode.
    effective_fill = fill
    cmap_resolved: Optional[Any] = None
    if is_2d and cbar and hue is None:
        if effective_fill is None:
            effective_fill = True
        cmap_resolved = cmap if cmap is not None else plt.rcParams["image.cmap"]
        sns_kwargs["cmap"] = cmap_resolved

    if effective_fill is not None:
        sns_kwargs["fill"] = effective_fill
    if hue is None:
        # Don't pass `color` when we're letting `cmap` drive the bands;
        # seaborn would otherwise build a discrete light_palette from
        # `color` and re-discretize the cmap.
        if color is not None and cmap_resolved is None:
            sns_kwargs["color"] = color
    else:
        sns_kwargs["palette"] = palette_map if palette_map else palette
    sns_kwargs.update(kwargs)

    sns.kdeplot(**sns_kwargs)

    # ---- 1D face/edge double-layer paint ----------------------------------
    # Mirror hist.py::_paint_lines: lines carry the opaque edge color, filled
    # PolyCollections get alpha on face and 1.0 on edge. For 1D only — 2D
    # contour collections carry per-level colormap array data that we must
    # not overwrite.
    if not is_2d:
        _paint_1d(
            new_lines=tracker.get_new_lines(),
            new_collections=tracker.get_new_collections(),
            palette=palette_map,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    else:
        # 2D contour sets need linewidth applied post-hoc (seaborn's
        # contour path rejects the kwarg upstream). Face/edge colors are
        # driven by the QuadContourSet's cmap so we leave those alone.
        for collection in tracker.get_new_collections():
            try:
                collection.set_linewidth(linewidth)
            except AttributeError:
                pass

    # ---- Labels ----------------------------------------------------------
    # None → preserve seaborn's inferred label (column name / 'Density').
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # ---- Legend stash ----------------------------------------------------
    # Merge cbar_kws into legend_kws: cbar_kws is the seaborn-compat name for
    # colorbar-specific options; legend_kws is our canonical kwarg bucket.
    merged_legend_kws: Dict = dict(legend_kws or {})
    if cbar_kws:
        for k, v in cbar_kws.items():
            merged_legend_kws.setdefault(k, v)

    _legend(
        ax=ax,
        hue=hue,
        palette=palette_map,
        is_2d=is_2d,
        fill=fill,
        multiple=multiple,
        cbar=cbar,
        cbar_cmap=cmap_resolved,
        new_collections=tracker.get_new_collections(),
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        color=color,
        legend=legend,
        legend_kws=merged_legend_kws,
    )

    return ax


# =============================================================================
# Helper Functions
# =============================================================================


def _match_line_to_level(line: Line2D, palette: Dict) -> Optional[str]:
    """Identify the hue level whose palette entry matches the line color."""
    sample = to_rgba(line.get_color())[:3]
    best = None
    best_dist = float("inf")
    for level, col in palette.items():
        target = to_rgba(col)[:3]
        dist = sum((a - b) ** 2 for a, b in zip(sample, target))
        if dist < best_dist:
            best_dist = dist
            best = level
    return best


def _match_collection_to_level(
    collection: PolyCollection,
    palette: Dict,
) -> Optional[str]:
    """Identify the hue level whose palette entry matches the face color."""
    fc = collection.get_facecolor()
    if len(fc) == 0:
        ec = collection.get_edgecolor()
        if len(ec) == 0:
            return None
        sample = ec[0][:3]
    else:
        sample = fc[0][:3]
    best = None
    best_dist = float("inf")
    for level, col in palette.items():
        target = to_rgba(col)[:3]
        dist = sum((a - b) ** 2 for a, b in zip(sample, target))
        if dist < best_dist:
            best_dist = dist
            best = level
    return best


def _paint_1d(
    new_lines: List[Line2D],
    new_collections: List,
    palette: Optional[Dict],
    color: Optional[str],
    edgecolor: Optional[str],
    linewidth: float,
    alpha: float,
) -> None:
    """Apply publiplots' transparent-fill / opaque-edge split to 1D KDE.

    With ``fill=False`` seaborn emits a ``Line2D`` per hue level; with
    ``fill=True`` (or stacked/fill multiple) it emits
    ``FillBetweenPolyCollection`` / ``PolyCollection`` objects. Both are
    recolored from the resolved palette (when hue is present) and the
    publiplots alpha-on-face / 1.0-on-edge split is applied.
    """
    # Line2D outlines — always opaque, full palette color.
    for line in new_lines:
        if palette:
            level = _match_line_to_level(line, palette)
            line_color = palette.get(level, color) if level is not None else color
        else:
            line_color = color
        if line_color is not None:
            line.set_color(to_rgba(line_color, alpha=1.0))
        line.set_linewidth(linewidth)

    # Filled density regions — alpha on face, opaque on edge.
    for collection in new_collections:
        if not isinstance(collection, PolyCollection):
            continue
        if palette:
            level = _match_collection_to_level(collection, palette)
            face_color = palette.get(level, color) if level is not None else color
        else:
            face_color = color
        if face_color is None:
            continue
        edge = edgecolor if edgecolor is not None else face_color
        collection.set_facecolor(to_rgba(face_color, alpha=alpha))
        collection.set_edgecolor(to_rgba(edge, alpha=1.0))
        collection.set_linewidth(linewidth)


def _find_quadcontour(new_collections: List):
    """Return the first QuadContourSet among new_collections, or None.

    QuadContourSet carries the cmap + data array used to derive a
    meaningful colorbar normalization (seaborn's ``NoNorm`` is a
    per-level index, not a density scale).
    """
    from matplotlib.contour import QuadContourSet
    for c in new_collections:
        if isinstance(c, QuadContourSet):
            return c
    return None


def _derive_cbar_norm(contour_set) -> Normalize:
    """Build a Normalize spanning the contour set's density levels.

    Seaborn sets ``norm=NoNorm()`` on the QuadContourSet so each
    contour band picks a discrete cmap index; that's the wrong scale
    for a publishable colorbar, which should read density. Rebuilding
    from ``contour_set.levels`` (the actual density cutoffs) gives a
    continuous mapping that matches what the contours encode.
    """
    levels = np.asarray(getattr(contour_set, "levels", []), dtype=float)
    if levels.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    return Normalize(vmin=float(np.min(levels)), vmax=float(np.max(levels)))


def _legend(
    ax: Axes,
    hue: Optional[str],
    palette: Optional[Dict],
    is_2d: bool,
    fill: Optional[bool],
    multiple: str,
    cbar: bool,
    cbar_cmap: Optional[Any],
    new_collections: List,
    alpha: float,
    linewidth: float,
    edgecolor: Optional[str],
    color: Optional[str],
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
) -> None:
    """Stash the mode-specific LegendEntry and render per-axis.

    Stashing logic (mirrors the spec):

    - 1D, hue=None: nothing stashed.
    - 1D + categorical hue: one ``"hue"`` entry. Rectangle handles when
      the drawn fill is active, line handles otherwise. Effective fill
      derived from seaborn's own default when the user passed
      ``fill=None``.
    - 2D, hue=None, cbar=True: one continuous-hue entry wrapping a
      ScalarMappable built from the QuadContourSet's cmap + density
      range.
    - 2D, hue=None, cbar=False: nothing stashed.
    - 2D + categorical hue: one ``"hue"`` entry with line handles per
      group.
    """
    if legend is False:
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})

    # Nothing hue-like to stash → render any existing entries and exit.
    if hue is None:
        if is_2d and cbar and flags["hue"]:
            contour_set = _find_quadcontour(new_collections)
            if contour_set is not None:
                hue_label = legend_kws.pop("hue_label", "density")
                # Prefer the explicitly resolved cmap from the call site
                # (a continuous Colormap). The QuadContourSet's own
                # cmap is a discrete light_palette built from `color`
                # and only has N≈levels swatches — wrong for a smooth
                # colorbar gradient.
                cmap = cbar_cmap
                if cmap is None:
                    cmap = contour_set.get_cmap() or plt.rcParams["image.cmap"]
                stash_continuous_hue(
                    ax,
                    name=hue_label,
                    palette=cmap,
                    hue_norm=_derive_cbar_norm(contour_set),
                )
        render_entries(ax, flags=flags, legend_kws=legend_kws)
        return

    # hue is set — need a palette dict to build categorical entries.
    if not palette:
        render_entries(ax, flags=flags, legend_kws=legend_kws)
        return

    hue_label = legend_kws.pop("hue_label", hue)

    # Derive effective fill for legend-handle style.
    # Seaborn defaults: fill=False for multiple='layer', fill=True for
    # 'stack'/'fill'. If user passed fill explicitly, honor it.
    if fill is None:
        effective_fill = multiple in ("stack", "fill")
    else:
        effective_fill = bool(fill)

    if flags["hue"]:
        labels = list(palette.keys())
        colors = [palette[v] for v in labels]
        edgecolors = [edgecolor] * len(labels) if edgecolor else None

        if is_2d:
            # 2D + categorical hue: line handles (per-group contour color).
            handles = create_legend_handles(
                labels=labels,
                colors=colors,
                edgecolors=edgecolors,
                linestyles=["-"] * len(labels),
                alpha=alpha,
                linewidth=linewidth,
            )
        elif effective_fill:
            handles = create_legend_handles(
                labels=labels,
                colors=colors,
                edgecolors=edgecolors,
                alpha=alpha,
                linewidth=linewidth,
                style="rectangle",
            )
        else:
            handles = create_legend_handles(
                labels=labels,
                colors=colors,
                edgecolors=edgecolors,
                linestyles=["-"] * len(labels),
                alpha=alpha,
                linewidth=linewidth,
            )

        stash_entry(
            ax,
            LegendEntry.build(
                name=hue_label,
                kind="hue",
                handles=handles,
                labels=labels,
            ),
        )

    render_entries(ax, flags=flags, legend_kws=legend_kws)
