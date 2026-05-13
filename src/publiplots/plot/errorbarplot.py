"""
Errorbar plot for scatter measurements with x and/or y uncertainty.

:func:`errorbarplot` is a 2D scatter primitive that overlays uncertainty
stems on each marker. Marker rendering — including hue routing,
palette selection, the publiplots face/edge alpha split, opaque
background marker for occlusion, and continuous-hue colorbar stashing
— is delegated to :func:`pp.scatterplot`. Errorbar stems are layered
*below* the markers (``zorder=1`` vs the scatter's ``zorder=2``).

Stems match the marker color: categorical ``hue=`` issues one stem
group per level (``ecolor=palette[level]``); numeric ``hue=`` issues
one stem per point (``ecolor=cmap(norm(value))``); no ``hue=`` uses
the neutral ``rcParams['edgecolor']``. Caller-supplied
``errorbar_kws['ecolor']`` always wins.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from publiplots.plot.scatter import scatterplot as _pp_scatter
from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.rcparams import resolve_param
from publiplots.utils import is_categorical


def errorbarplot(
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
    background_marker: Union[bool, str] = True,
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
    below the marker.

    **Stem coloring.** Stems match the marker color so the uncertainty
    visually attaches to the measurement:

    - **Categorical ``hue=``**: stems issued per-group with ``ecolor``
      set to the group's palette color (one ``ax.errorbar`` call per
      level).
    - **Continuous numeric ``hue=``**: stems issued per-point with
      ``ecolor`` resolved from the cmap and norm (one ``ax.errorbar``
      call per row — acceptable since errorbar plots typically show
      <100 points).
    - **No ``hue=``**: single ``ax.errorbar`` call with
      ``ecolor=rcParams['edgecolor']``.
    - **``errorbar_kws={'ecolor': ...}``** always wins over the above.
      Useful when a uniform stem color is wanted regardless of hue
      (e.g. neutral stems on a categorical hue plot).

    **Layering.** Errorbar stems use ``zorder=1``; scatter markers use
    ``zorder=2`` (the default from :func:`pp.scatterplot`). The marker
    is drawn last so the stem appears to terminate at the marker edge.
    ``background_marker=True`` (the default) draws an opaque white
    twin behind each marker so the stem does not show through the
    alpha-faded marker face.

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
        Column name for marker color grouping. Categorical values
        produce per-group stem colors via ``palette``; numeric values
        produce a continuous colorbar with per-point stem colors
        resolved from the cmap.
    palette : str, dict, or list, optional
        Color palette for hue values. See :func:`pp.scatterplot`.
    color : str, optional
        Fixed marker color when ``hue`` is None. Stems pick up this
        color too.
    linewidth : float, optional
        Stroke width for both marker edges and errorbar stems. Falls
        back to ``rcParams['lines.linewidth']``.
    edgecolor : str, optional
        Color for marker edges *and* the default stem color when
        ``hue`` is None or continuous. Falls back to
        ``rcParams['edgecolor']``. Override the stem color
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
    background_marker : bool or str, default True
        Draw an opaque background twin behind each marker so the stem
        does not show through the alpha-faded marker face. ``True``
        uses white; pass a color string (e.g. ``"#eeeeee"``) to match
        a non-white axes background. Set ``False`` to allow the stem
        to read through the marker.
    errorbar_kws : dict, optional
        Extra keyword arguments forwarded to
        :meth:`matplotlib.axes.Axes.errorbar`. Common overrides:

        - ``ecolor`` — stem color. Wins over the hue-derived color.
          Useful when you want neutral stems even with a categorical
          hue (pass ``ecolor=pp.rcParams['edgecolor']``).
        - ``elinewidth`` — stem thickness (defaults to ``linewidth``).
        - ``capsize``, ``capthick`` — see top-level kwargs.

        Caller-supplied keys win via ``setdefault``.
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to :func:`pp.scatterplot`.
        ``background_marker`` is forwarded automatically; pass it here
        only if you want to override the top-level kwarg per-call.
    ax : Axes, optional
        Target axes. When ``None``, a new figure is created via
        :func:`pp.subplots`.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    legend : bool or dict, default True
        Legend control passed through to :func:`pp.scatterplot`.
        ``False`` stashes nothing. A dict maps legend kinds to
        booleans.
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

    >>> ax = pp.errorbarplot(data=df, x="dose", y="response", yerr="sem")

    Both directions:

    >>> ax = pp.errorbarplot(data=df, x="dose", y="response",
    ...                       xerr="dose_err", yerr="response_err")

    Asymmetric y-errors via a (lo, hi) column tuple:

    >>> ax = pp.errorbarplot(data=df, x="dose", y="response",
    ...                       yerr=("ci_lo", "ci_hi"))

    Constant scalar error:

    >>> ax = pp.errorbarplot(data=df, x="x", y="y", xerr=0.1, yerr=0.2)

    Hue groups with per-group stem colors:

    >>> ax = pp.errorbarplot(data=df, x="dose", y="response", yerr="sem",
    ...                       hue="treatment", palette="pastel")

    Continuous hue with colorbar legend (neutral stems):

    >>> ax = pp.errorbarplot(data=df, x="x", y="y", yerr="err",
    ...                       hue="score", palette="viridis")

    Visible caps:

    >>> ax = pp.errorbarplot(data=df, x="x", y="y", yerr="err", capsize=2)

    Force neutral stems even with a categorical hue:

    >>> ax = pp.errorbarplot(data=df, x="x", y="y", yerr="err",
    ...                       hue="g", errorbar_kws={"ecolor": "0.4"})

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

    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    edgecolor = resolve_param("edgecolor", edgecolor)
    capsize = resolve_param("capsize", capsize)
    capthick_resolved = capthick if capthick is not None else linewidth

    _validate_err_columns(xerr, "xerr", data)
    _validate_err_columns(yerr, "yerr", data)

    if ax is None:
        _, ax = _pp_subplots()

    # Default background_marker=True so the alpha-faded marker face
    # doesn't reveal the stem behind it. Caller can opt out per-call
    # via top-level kwarg or via scatter_kws['background_marker'].
    scatter_kws = dict(scatter_kws or {})
    scatter_kws.setdefault("background_marker", background_marker)

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

    if xerr_arr is None and yerr_arr is None:
        if title is not None:  ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        return ax

    ekws_user = dict(errorbar_kws or {})
    ekws_user.setdefault("elinewidth", linewidth)
    ekws_user.setdefault("capsize", capsize)
    ekws_user.setdefault("capthick", capthick_resolved)

    # Stem color routing:
    #   - ecolor in user errorbar_kws -> single call, that color
    #     (caller wins; lets users force neutral stems on hue plots)
    #   - categorical hue -> per-group call with ecolor=palette[level]
    #   - continuous hue -> per-point call with ecolor=cmap(norm(value))
    #     (one ax.errorbar call per row; only choice that matches stems
    #     to the marker cmap. Acceptable since errorbar plots typically
    #     have <100 points.)
    #   - else (no hue) -> single call, ecolor=edgecolor
    user_ecolor = ekws_user.pop("ecolor", None)

    if user_ecolor is not None or hue is None:
        ax.errorbar(
            np.asarray(data[x]),
            np.asarray(data[y]),
            xerr=xerr_arr,
            yerr=yerr_arr,
            fmt="none",
            zorder=1,
            ecolor=user_ecolor if user_ecolor is not None else edgecolor,
            **ekws_user,
        )
    elif is_categorical(data[hue]):
        levels = data[hue].unique()
        palette_map = resolve_palette_map(values=levels, palette=palette)
        for level in levels:
            mask = (data[hue] == level).to_numpy()
            xs = np.asarray(data[x])[mask]
            ys = np.asarray(data[y])[mask]
            xerr_g = _slice_err(xerr_arr, mask)
            yerr_g = _slice_err(yerr_arr, mask)
            ax.errorbar(
                xs, ys,
                xerr=xerr_g, yerr=yerr_g,
                fmt="none",
                zorder=1,
                ecolor=palette_map[level],
                **ekws_user,
            )
    else:
        # Continuous numeric hue. Resolve cmap+norm directly from the
        # hue column (NOT from the foreground PathCollection's face
        # colors — those have been composited over the background
        # marker's white at the user's alpha, so reading them back
        # would give washed-out stems).
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        values = np.asarray(data[hue], dtype=float)
        norm = hue_norm
        if norm is None:
            norm = Normalize(vmin=values.min(), vmax=values.max())
        elif isinstance(norm, tuple):
            norm = Normalize(vmin=norm[0], vmax=norm[1])
        cmap = colormaps[palette] if isinstance(palette, str) else palette
        stem_colors = cmap(norm(values))

        xs = np.asarray(data[x])
        ys = np.asarray(data[y])
        for i in range(len(data)):
            xerr_i = _slice_err(xerr_arr, np.array([i]))
            yerr_i = _slice_err(yerr_arr, np.array([i]))
            ax.errorbar(
                xs[i:i + 1], ys[i:i + 1],
                xerr=xerr_i, yerr=yerr_i,
                fmt="none",
                zorder=1,
                ecolor=stem_colors[i],
                **ekws_user,
            )

    if title is not None:  ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)

    return ax


def _validate_err_columns(err, name: str, data: pd.DataFrame) -> None:
    if err is None:
        return
    if isinstance(err, tuple) and len(err) == 2 and all(isinstance(e, str) for e in err):
        missing = [c for c in err if c not in data.columns]
        if missing:
            raise ValueError(f"{name} columns not found in data: {missing}")
        return
    if isinstance(err, str):
        if err not in data.columns:
            raise ValueError(f"{name} column '{err}' not found in data.")


def _resolve_err(err, data: pd.DataFrame):
    if err is None:
        return None
    if isinstance(err, tuple) and len(err) == 2 and all(isinstance(e, str) for e in err):
        lo = np.asarray(data[err[0]].to_numpy(), dtype=float)
        hi = np.asarray(data[err[1]].to_numpy(), dtype=float)
        return np.vstack([lo, hi])
    if isinstance(err, str):
        return np.asarray(data[err].to_numpy(), dtype=float)
    return np.asarray(err)


def _slice_err(err_arr, mask):
    """Slice a resolved err array along the row axis using a boolean mask.

    Handles the four shapes ``_resolve_err`` produces:
      - ``None`` -> ``None``
      - ``(N,)`` -> ``arr[mask]``
      - ``(2, N)`` -> ``arr[:, mask]``
      - scalar -> unchanged (broadcast at draw time)
    """
    if err_arr is None:
        return None
    arr = np.asarray(err_arr)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 2:
        return arr[:, mask]
    return arr[mask]
