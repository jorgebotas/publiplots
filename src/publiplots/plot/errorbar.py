"""
Errorbar plot for scatter measurements with x and/or y uncertainty.

:func:`errorbar` is a 2D scatter primitive that overlays uncertainty
stems on each marker. Marker rendering — including hue routing,
palette selection, the publiplots face/edge alpha split, and continuous-
hue colorbar stashing — is delegated to :func:`pp.scatterplot`. The
errorbar stems themselves are drawn with a single
``ax.errorbar(fmt='none', ...)`` call layered *below* the markers
(``zorder=1`` vs the scatter's ``zorder=2``).

Stems default to ``rcParams['edgecolor']`` rather than the per-point
hue color: a neutral, opaque stem reads as "uncertainty" on top of any
qualitative or continuous colormapping without competing for attention.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from publiplots.plot.scatter import scatterplot as _pp_scatter
from publiplots.themes.rcparams import resolve_param


def errorbar(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    xerr: Optional[Union[str, float, Sequence, Tuple[str, str]]] = None,
    yerr: Optional[Union[str, float, Sequence, Tuple[str, str]]] = None,
    hue: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    alpha: Optional[float] = None,
    capsize: Optional[float] = None,
    capthick: Optional[float] = None,
    hue_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    errorbar_kws: Optional[Dict] = None,
    scatter_kws: Optional[Dict] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs,
) -> Axes:
    """
    Create a publication-ready 2D errorbar plot: scatter with uncertainty stems.

    Each ``(x, y)`` pair is drawn as a marker (via :func:`pp.scatterplot`,
    inheriting all of its hue / palette / face-edge alpha behavior) with
    optional x- and/or y-direction uncertainty rendered as thin stems
    below the marker. Stems are issued in a single
    :meth:`matplotlib.axes.Axes.errorbar` call regardless of hue grouping
    — they share a uniform neutral color (``rcParams['edgecolor']``) so
    they don't compete with hue-mapped marker faces for visual weight.

    Errorbars use ``zorder=1``; scatter markers use ``zorder=2`` (the
    default from :func:`pp.scatterplot`). The marker face is drawn last
    so the stem appears to terminate at the marker edge.

    Parameters
    ----------
    data : DataFrame
        Input data containing ``x``, ``y``, and any error / hue columns.
    x, y : str
        Column names for the bivariate axes.
    xerr, yerr : str, float, sequence, or (str, str) tuple, optional
        Uncertainty in the x- and/or y- direction:

        - ``str`` — column name; used as a symmetric ``±err`` per row.
        - ``float`` — constant scalar; broadcast to every row.
        - ``(lo_col, hi_col)`` — two column names: lower and upper
          asymmetric extents (each ``≥ 0``). Renders as
          ``[y - lo, y + hi]``.
        - ``Sequence`` — array-like with shape ``(N,)`` or ``(2, N)``,
          forwarded directly to :meth:`Axes.errorbar`.

        ``None`` (the default) suppresses stems in that direction. When
        both ``xerr`` and ``yerr`` are ``None`` no errorbar call is
        issued at all.
    hue : str, optional
        Column name for marker color grouping. Categorical or
        continuous, forwarded verbatim to :func:`pp.scatterplot`.
    palette : str, dict, or list, optional
        Color palette for hue values. See :func:`pp.scatterplot`.
    color : str, optional
        Fixed marker color when ``hue`` is None.
    linewidth : float, optional
        Stroke width for both marker edges and errorbar stems. Falls
        back to ``rcParams['lines.linewidth']``.
    edgecolor : str, optional
        Color for marker edges *and* the default stem color. Falls back
        to ``rcParams['edgecolor']``. Override the stem color
        independently via ``errorbar_kws={'ecolor': ...}``.
    alpha : float, optional
        Marker face transparency (0–1). Edges stay opaque (publiplots
        double-layer convention). Stems are unaffected — they use the
        opaque ``ecolor``. Falls back to ``rcParams['alpha']``.
    capsize : float, optional
        Half-length of the stem caps in points. Falls back to
        ``rcParams['capsize']`` (defaults to ``0`` — no caps).
    capthick : float, optional
        Stroke width of the stem caps. Defaults to ``linewidth``.
    hue_norm : (vmin, vmax) or Normalize, optional
        Color normalization for continuous hue. Forwarded to
        :func:`pp.scatterplot`.
    errorbar_kws : dict, optional
        Extra keyword arguments forwarded to
        :meth:`matplotlib.axes.Axes.errorbar`. Common overrides:

        - ``ecolor`` — stem color (defaults to ``edgecolor``).
        - ``elinewidth`` — stem thickness (defaults to ``linewidth``).
        - ``capsize``, ``capthick`` — see top-level kwargs.

        Caller-supplied keys win via ``setdefault``.
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to :func:`pp.scatterplot`.
    ax : Axes, optional
        Target axes. When ``None``, a new figure is created via
        :func:`pp.subplots`.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels. ``None`` (the default) preserves seaborn's
        column-name inference from :func:`pp.scatterplot`.
    legend : bool or dict, default True
        Legend control passed through to :func:`pp.scatterplot`.
        ``False`` stashes nothing. A dict maps legend kinds (``"hue"``,
        ``"size"``, ``"style"``) to booleans.
    legend_kws : dict, optional
        Forwarded to the legend builder; supports e.g. ``hue_label``
        for overriding the legend title.
    **kwargs
        Reserved for the ``figsize`` rejection guard. ``figsize`` is
        rejected — use ``pp.subplots(axes_size=...)`` and pass ``ax=``
        instead. No other kwargs are forwarded.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Notes
    -----
    Marker shape is ``"o"`` (the :func:`pp.scatterplot` default). The
    marker size comes from ``rcParams['lines.markersize']`` — pass a
    custom ``s=`` via ``scatter_kws`` to override.

    Examples
    --------
    Symmetric y-error bars:

    >>> ax = pp.errorbar(data=df, x="dose", y="response", yerr="sem")

    Both directions:

    >>> ax = pp.errorbar(data=df, x="dose", y="response",
    ...                   xerr="dose_err", yerr="response_err")

    Asymmetric y-errors via a (lo, hi) column tuple:

    >>> ax = pp.errorbar(data=df, x="dose", y="response",
    ...                   yerr=("ci_lo", "ci_hi"))

    Constant scalar error:

    >>> ax = pp.errorbar(data=df, x="x", y="y", xerr=0.1, yerr=0.2)

    Hue groups with neutral stems:

    >>> ax = pp.errorbar(data=df, x="dose", y="response", yerr="sem",
    ...                   hue="treatment", palette="pastel")

    Continuous hue with colorbar legend:

    >>> ax = pp.errorbar(data=df, x="x", y="y", yerr="err",
    ...                   hue="score", palette="viridis")

    Visible caps:

    >>> ax = pp.errorbar(data=df, x="x", y="y", yerr="err", capsize=2)

    See Also
    --------
    publiplots.scatterplot : Marker layer used internally (no stems).
    publiplots.pointplot : Aggregated point estimate with bootstrap CI.
    """
    from publiplots.layout.subplots import (
        reject_figsize,
        subplots as _pp_subplots,
    )
    reject_figsize(kwargs)

    # rcParams resolution: same as pp.scatterplot for marker styling
    # plus the publiplots-specific 'capsize' default.
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    edgecolor = resolve_param("edgecolor", edgecolor)
    capsize = resolve_param("capsize", capsize)
    capthick_resolved = capthick if capthick is not None else linewidth

    # Validate xerr / yerr columns up front so we surface clear messages.
    # (pp.scatterplot's own validation only sees x, y, hue.)
    _validate_err_columns(xerr, "xerr", data)
    _validate_err_columns(yerr, "yerr", data)

    if ax is None:
        _, ax = _pp_subplots()

    # Delegate marker rendering. pp.scatterplot already handles:
    #   - hue routing (categorical -> palette, continuous -> colorbar)
    #   - face/edge alpha split (face=alpha, edge=1.0)
    #   - LegendEntry stashing (hue, size, style)
    #   - continuous-hue ScalarMappable for the colorbar legend.
    scatter_kws = dict(scatter_kws or {})
    _pp_scatter(
        data=data, x=x, y=y,
        hue=hue, palette=palette, color=color,
        linewidth=linewidth, edgecolor=edgecolor, alpha=alpha,
        hue_norm=hue_norm,
        ax=ax, legend=legend, legend_kws=legend_kws,
        **scatter_kws,
    )

    xerr_arr = _resolve_err(xerr, data)
    yerr_arr = _resolve_err(yerr, data)

    if xerr_arr is not None or yerr_arr is not None:
        ekws = dict(errorbar_kws or {})
        # Stems use the neutral edgecolor by default — never the per-point
        # hue color — so they read as "uncertainty" rather than competing
        # with the marker face.
        ekws.setdefault("ecolor", edgecolor)
        ekws.setdefault("elinewidth", linewidth)
        ekws.setdefault("capsize", capsize)
        ekws.setdefault("capthick", capthick_resolved)
        ax.errorbar(
            np.asarray(data[x]),
            np.asarray(data[y]),
            xerr=xerr_arr,
            yerr=yerr_arr,
            fmt="none",        # no marker — pp.scatterplot drew them.
            zorder=1,          # below scatter (zorder=2 from pp.scatterplot).
            **ekws,
        )

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax


def _validate_err_columns(err, name: str, data: pd.DataFrame) -> None:
    """Raise a clear ValueError if ``err`` references a missing column.

    Accepts the same shapes as :func:`_resolve_err`:

    - ``str`` — must exist in ``data.columns``.
    - ``(lo, hi)`` tuple of two strings — both must exist.
    - scalar / sequence / None — pass through unchanged.
    """
    if err is None:
        return
    if isinstance(err, tuple) and len(err) == 2 and all(isinstance(e, str) for e in err):
        missing = [c for c in err if c not in data.columns]
        if missing:
            raise ValueError(
                f"{name} columns not found in data: {missing}"
            )
        return
    if isinstance(err, str):
        if err not in data.columns:
            raise ValueError(
                f"{name} column '{err}' not found in data."
            )


def _resolve_err(err, data: pd.DataFrame):
    """Convert the user's ``xerr`` / ``yerr`` argument to the array form
    expected by :meth:`Axes.errorbar`.

    - ``None`` -> ``None`` (suppresses stems in that direction).
    - ``str`` -> 1-D numpy array from ``data[col]`` (symmetric ±err).
    - ``(lo_col, hi_col)`` -> 2-row numpy array (asymmetric).
    - scalar / sequence -> coerced via ``np.asarray`` (matplotlib handles
      broadcasting from scalar / 1-D / 2-row inputs).
    """
    if err is None:
        return None
    if isinstance(err, tuple) and len(err) == 2 and all(isinstance(e, str) for e in err):
        lo = np.asarray(data[err[0]].to_numpy(), dtype=float)
        hi = np.asarray(data[err[1]].to_numpy(), dtype=float)
        return np.vstack([lo, hi])
    if isinstance(err, str):
        return np.asarray(data[err].to_numpy(), dtype=float)
    return np.asarray(err)
