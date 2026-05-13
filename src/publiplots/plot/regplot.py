"""
Regression plot for publiplots.

:func:`regplot` wraps :func:`seaborn.regplot` with publiplots styling and
adds a crucial missing feature: categorical ``hue=`` support. Seaborn's
native ``regplot`` does not accept ``hue``; users historically had to
reach for ``sns.lmplot`` (which owns its figure) to get per-group fits.
``pp.regplot`` loops over hue levels and calls ``sns.regplot`` per group
onto a shared axes, preserving compatibility with :func:`pp.subplots`
and the figure/axes layout pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.rcparams import resolve_param
from publiplots.utils import create_legend_handles, is_numeric
from publiplots.utils.legend_entries import (
    LegendEntry,
    resolve_legend_flags,
    stash_entry,
)
from publiplots.utils.plot_legend import render_entries


def regplot(
    data: Optional[pd.DataFrame] = None,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    marker: Optional[str] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    x_estimator: Optional[Any] = None,
    x_bins: Optional[Any] = None,
    x_ci: Union[str, int, None] = "ci",
    scatter: bool = True,
    fit_reg: bool = True,
    ci: Optional[int] = 95,
    n_boot: int = 1000,
    units: Optional[str] = None,
    seed: Optional[int] = None,
    order: int = 1,
    logistic: bool = False,
    lowess: bool = False,
    robust: bool = False,
    logx: bool = False,
    x_partial: Optional[Any] = None,
    y_partial: Optional[Any] = None,
    truncate: bool = True,
    dropna: bool = True,
    x_jitter: Optional[float] = None,
    y_jitter: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    scatter_kws: Optional[Dict] = None,
    line_kws: Optional[Dict] = None,
    ci_kws: Optional[Dict] = None,
    **kwargs,
) -> Axes:
    """
    Create a publication-ready regression plot with optional hue grouping.

    Wraps :func:`seaborn.regplot` with the publiplots styling pipeline and
    adds ``hue=`` support (categorical only). When ``hue`` is provided the
    plot loops over hue levels and draws a separate regression per group
    onto the same axes.

    The plot has three independent visual layers, each with its own
    styling bucket:

    +-------------+-------------------------+-------------------------+
    | Layer       | Artist                  | Styling kwarg           |
    +=============+=========================+=========================+
    | scatter     | ``PathCollection``      | ``scatter_kws``         |
    +-------------+-------------------------+-------------------------+
    | fit line    | ``Line2D``              | ``line_kws``            |
    +-------------+-------------------------+-------------------------+
    | CI band     | ``FillBetweenPolyColl.``| ``ci_kws``              |
    +-------------+-------------------------+-------------------------+

    The top-level ``alpha``, ``edgecolor``, ``linewidth``, ``marker`` and
    ``color`` kwargs are convenience shortcuts that publiplots routes to
    the appropriate layer's bucket via ``setdefault`` (so anything you
    pass directly in ``scatter_kws`` / ``line_kws`` / ``ci_kws`` always
    wins).

    Unlike seaborn's ``regplot``, the ``alpha`` knob applies only to the
    scatter face — edges stay opaque (the publiplots double-layer
    convention shared by ``pp.scatterplot``, ``pp.histplot``, etc.). The
    fit line stays at full opacity, and the CI band keeps seaborn's
    light alpha unless overridden via ``ci_kws['alpha']``.

    Parameters
    ----------
    data : DataFrame
        Input dataset.
    x, y : str
        Column names for the bivariate axes.
    hue : str, optional
        Column name for categorical color grouping. A separate regression
        is drawn per hue level. Continuous (numeric) hue columns emit a
        :class:`UserWarning` and fall back to a single regression.
    palette : str, dict, or list, optional
        Color palette. Accepts a palette name (publiplots or seaborn), an
        explicit list of colors, or a ``{hue_level: color}`` dict.
    color : str, optional
        Fixed color used when ``hue`` is None.
    alpha : float, optional
        Transparency level for scatter markers (0-1). Falls back to
        ``rcParams["alpha"]``.
    marker : str, optional
        Matplotlib marker code forwarded to seaborn (e.g. ``"s"``,
        ``"^"``). Do **not** pass ``marker`` via ``scatter_kws`` — seaborn
        hard-overwrites the scatter collection's paths with whatever it
        received at the top level.
    linewidth : float, optional
        Width of marker edges (and fit-line thickness via ``line_kws``
        default). Falls back to ``rcParams["lines.linewidth"]``.
    edgecolor : str, optional
        Color for scatter marker edges. Set via ``scatter_kws["edgecolor"]``
        when the user wants per-call control; the top-level ``edgecolor=``
        is a convenience shortcut.
    x_estimator, x_bins, x_ci : optional
        See :func:`seaborn.regplot`. When ``x_estimator`` is provided, the
        scatter is replaced by a per-bin aggregated point with optional
        error bars.
    scatter : bool, default=True
        Draw the scatter layer.
    fit_reg : bool, default=True
        Draw the regression line + confidence band.
    ci : int, default=95
        Confidence interval for the regression band, in percent. ``None``
        disables the band.
    n_boot : int, default=1000
        Bootstrap iterations for the CI band.
    units, seed, order, logistic, lowess, robust, logx, x_partial, y_partial, truncate, dropna, x_jitter, y_jitter : optional
        See :func:`seaborn.regplot`.
    ax : Axes, optional
        Matplotlib axes to draw on. If None, a new figure is created via
        :func:`publiplots.layout.subplots`.
    title : str, optional
        Plot title. ``None`` leaves the axes title untouched.
    xlabel, ylabel : str, optional
        Axis labels. ``None`` (the default) preserves whatever seaborn
        inferred from the data (usually the column names).
    legend : bool or dict, default=True
        Legend control. ``True`` stashes and renders a hue legend when
        applicable. ``False`` hides it. A dict enables/disables specific
        kinds (e.g. ``{"hue": False}``).
    legend_kws : dict, optional
        Additional kwargs for the legend builder. Supported keys include
        ``hue_label`` to override the title.
    scatter_kws : dict, optional
        Extra kwargs forwarded to :meth:`matplotlib.axes.Axes.scatter`
        via seaborn. Publiplots supplies sensible defaults for
        ``linewidths``, ``edgecolor``, and ``alpha`` via ``setdefault``,
        so caller-supplied values always win.
    line_kws : dict, optional
        Extra kwargs forwarded to the fit-line
        :meth:`matplotlib.axes.Axes.plot` call via seaborn.
    ci_kws : dict, optional
        Styling overrides for the confidence-interval band drawn
        around the fit (the ``FillBetweenPolyCollection`` seaborn
        emits when ``ci`` is not None). Recognized keys:

        - ``alpha`` : float — face alpha for the band. Default is
          seaborn's 0.15. Use a lower value to de-emphasize the band
          when overlaying multiple groups, or a higher value for a
          single bold fit.
        - ``color`` : color — face color. Defaults to the regression
          line color (typically the per-group palette entry under
          ``hue=``). Override when you want a band color independent
          of the line, e.g. for accessibility or print contrast.

        Applied per-group when ``hue=`` is set. ``ci_kws`` only
        affects the CI band; the regression line uses ``line_kws``,
        the scatter uses ``scatter_kws``.
    **kwargs
        Extra keyword arguments forwarded to :func:`seaborn.regplot`.
        ``figsize`` is rejected — use ``pp.subplots(axes_size=...)``
        and pass ``ax=`` instead.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Notes
    -----
    **Layer-specific styling**: prefer the per-layer ``*_kws`` buckets
    when you want fine-grained control. The top-level convenience
    kwargs only seed defaults via ``setdefault``:

    * scatter face transparency  → ``alpha`` (or ``scatter_kws['alpha']``)
    * scatter marker shape       → ``marker``
    * scatter edge stroke        → ``edgecolor`` + ``linewidth``
    * fit line                   → ``line_kws``  (e.g. ``{'linestyle': '--'}``)
    * CI band                    → ``ci_kws``    (e.g. ``{'alpha': 0.6}``)

    **statsmodels dependency**: ``lowess=True``, ``robust=True``, and
    ``logistic=True`` require ``statsmodels``. Install with
    ``pip install publiplots[regression]``. Without it, those flags
    raise ``RuntimeError`` from seaborn's ``_check_statsmodels``.

    **Hue handling**: only categorical hue is supported. Numeric hue
    columns emit ``UserWarning`` and fall back to a single-color
    regression — a colorbar would not meaningfully composite with N
    regression lines. Use ``pp.scatterplot(hue=, palette='viridis')``
    + a separate non-hued ``pp.regplot`` overlay if you need both.

    Examples
    --------
    Basic linear fit:

    >>> ax = pp.regplot(data=df, x="x", y="y")

    Polynomial fit:

    >>> ax = pp.regplot(data=df, x="x", y="y", order=2)

    Per-group fits via ``hue=`` (the novelty over :func:`seaborn.regplot`):

    >>> ax = pp.regplot(data=df, x="x", y="y", hue="group", palette="pastel")

    Binned aggregation with bootstrap CI:

    >>> ax = pp.regplot(
    ...     data=df, x="x", y="y",
    ...     x_bins=10, x_estimator=np.mean,
    ... )

    Bolder CI band (de-emphasize the regression line, emphasize the
    uncertainty):

    >>> ax = pp.regplot(data=df, x="x", y="y", ci_kws={"alpha": 0.6})

    De-emphasize CI bands when overlaying multiple groups (full-strength
    bands stack into mud):

    >>> ax = pp.regplot(
    ...     data=df, x="x", y="y", hue="group",
    ...     palette="pastel", ci_kws={"alpha": 0.1},
    ... )

    Decouple band color from line color (e.g. for accessibility / print
    contrast):

    >>> ax = pp.regplot(
    ...     data=df, x="x", y="y",
    ...     line_kws={"color": "navy"},
    ...     ci_kws={"color": "#888", "alpha": 0.3},
    ... )

    Custom marker, transparent face, opaque edge — full publiplots
    convention:

    >>> ax = pp.regplot(
    ...     data=df, x="x", y="y",
    ...     marker="^", alpha=0.4, edgecolor="black",
    ... )

    Suppress the regression line entirely (scatter + CI only — useful for
    showing uncertainty without committing to a fit visually):

    >>> ax = pp.regplot(data=df, x="x", y="y", line_kws={"alpha": 0})

    Or suppress just the CI band:

    >>> ax = pp.regplot(data=df, x="x", y="y", ci=None)

    See Also
    --------
    seaborn.regplot : Underlying rendering primitive (no hue support).
    publiplots.scatterplot : Scatter-only cousin with full hue/size/style.
    """
    from publiplots.layout.subplots import reject_figsize, subplots as _pp_subplots
    reject_figsize(kwargs)

    # Resolve rcParam defaults up front so both the hued and un-hued branches
    # see the same values.
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Validate required columns early; seaborn's own error message is cryptic
    # and we'd rather surface the missing column name clearly.
    if data is not None and isinstance(data, pd.DataFrame):
        required = [c for c in (x, y, hue) if c is not None]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

    if ax is None:
        fig, ax = _pp_subplots()

    # Hue handling: decide branch (single fit vs per-group loop).
    palette_map: Optional[Dict] = None
    numeric_hue_fallback = False
    if hue is not None and data is not None:
        if is_numeric(data[hue]):
            # Continuous hue is not supported on regplot — no colorbar would
            # meaningfully composite with N regression lines. Warn, then
            # fall through to the single-regression branch.
            warnings.warn(
                "continuous hue not supported on pp.regplot; "
                "falling back to single regression",
                UserWarning,
                stacklevel=2,
            )
            numeric_hue_fallback = True
        else:
            palette_map = resolve_palette_map(
                values=list(data[hue].unique()), palette=palette,
            )

    # Build the shared scatter_kws / line_kws. Using setdefault means the
    # caller's values win while publiplots fills in any missing keys.
    #
    # alpha is NOT pushed into scatter_kws — seaborn would call
    # collection.set_alpha(alpha) which applies uniformly to face AND
    # edge, breaking the publiplots double-layer convention (face
    # transparent, edge opaque). Instead we apply the face/edge split
    # post-draw via apply_transparency on the captured PathCollections.
    scatter_kws = dict(scatter_kws or {})
    scatter_kws.setdefault("linewidths", linewidth)
    if edgecolor is not None:
        scatter_kws.setdefault("edgecolor", edgecolor)

    line_kws = dict(line_kws or {})
    line_kws.setdefault("linewidth", linewidth)

    # Common kwargs for every sns.regplot call (both branches).
    base_kws: Dict[str, Any] = dict(
        x=x, y=y, ax=ax,
        x_estimator=x_estimator,
        x_bins=x_bins,
        x_ci=x_ci,
        scatter=scatter,
        fit_reg=fit_reg,
        ci=ci,
        n_boot=n_boot,
        units=units,
        seed=seed,
        order=order,
        logistic=logistic,
        lowess=lowess,
        robust=robust,
        logx=logx,
        x_partial=x_partial,
        y_partial=y_partial,
        truncate=truncate,
        dropna=dropna,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
        # marker is passed at the top level — seaborn hard-overwrites any
        # "marker" key inside scatter_kws, so we never stuff it there.
        marker=marker if marker is not None else "o",
    )
    base_kws.update(kwargs)

    # Capture every PathCollection the seaborn call(s) emit so we can
    # apply the publiplots face/edge alpha split after the draw.
    from publiplots.utils.transparency import ArtistTracker, apply_transparency
    tracker = ArtistTracker(ax)

    if palette_map is not None:
        # Per-group regression loop. Each call re-uses the same scatter_kws /
        # line_kws but with the per-group color — seaborn's own color param
        # wins over any color embedded in scatter_kws/line_kws.
        for level, group_color in palette_map.items():
            sub = data[data[hue] == level]
            if len(sub) == 0:
                continue
            sns.regplot(
                data=sub,
                color=group_color,
                scatter_kws=dict(scatter_kws),
                line_kws=dict(line_kws),
                **base_kws,
            )
    else:
        # Single-fit branch (no hue, or continuous-hue fallback).
        sns.regplot(
            data=data,
            color=color,
            scatter_kws=dict(scatter_kws),
            line_kws=dict(line_kws),
            **base_kws,
        )

    # Two post-draw passes over the new collections:
    #
    # 1. PathCollection (scatter): publiplots double-layer style — face
    #    carries alpha, edge stays opaque. We must clear the
    #    collection-level ``_alpha`` first; sns.regplot internally sets
    #    it via Collection.set_alpha(0.8) (its scatter default), and
    #    matplotlib's set_facecolors/set_edgecolors re-apply ``_alpha``
    #    after storing per-color RGBAs — so apply_transparency() would
    #    silently no-op without this reset.
    #
    # 2. FillBetweenPolyCollection (CI band): apply ci_kws overrides
    #    (alpha, color). Same _alpha-reset trick is needed for the
    #    same matplotlib reason. The regression line (Line2D, untouched
    #    here) keeps line_kws styling.
    from matplotlib.collections import PathCollection, FillBetweenPolyCollection
    ci_kws = dict(ci_kws or {})
    ci_alpha = ci_kws.get("alpha")
    ci_color = ci_kws.get("color")
    for c in tracker.get_new_collections():
        if isinstance(c, PathCollection):
            if alpha is not None:
                c.set_alpha(None)
                apply_transparency(c, face_alpha=alpha, edge_alpha=1.0)
        elif isinstance(c, FillBetweenPolyCollection):
            if ci_alpha is not None or ci_color is not None:
                # Recover the band's current face color (set by seaborn
                # to match the regression line) so we can preserve it
                # when only alpha is overridden.
                current_fc = np.asarray(c.get_facecolor())
                base_color = ci_color if ci_color is not None else (
                    tuple(current_fc[0][:3]) if current_fc.size else (0, 0, 0)
                )
                # Final alpha: explicit ci_kws['alpha'] wins; otherwise
                # preserve seaborn's existing alpha (typically 0.15).
                final_alpha = ci_alpha if ci_alpha is not None else (
                    float(current_fc[0][3]) if current_fc.size else 0.15
                )
                c.set_alpha(None)  # clear any collection-level alpha
                c.set_facecolor(to_rgba(base_color, alpha=final_alpha))

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    _legend(
        ax=ax,
        hue=hue,
        palette=palette_map,
        marker=marker,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        legend=legend,
        legend_kws=legend_kws,
    )

    return ax


def _legend(
    ax: Axes,
    hue: Optional[str],
    palette: Optional[Dict],
    marker: Optional[str],
    alpha: Optional[float],
    linewidth: Optional[float],
    edgecolor: Optional[str],
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
) -> None:
    """Stash a single hue LegendEntry (when applicable) and render.

    Follows the scatter semantics: one entry per hue level using
    marker-style handles. We do **not** stash an entry for the
    regression line itself — it shares the hue color, and the user
    already reads the line as "the fit for this group" when the
    scatter swatch is present.
    """
    # legend=False short-circuits both stash and render.
    if legend is False:
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})

    # No hue or numeric-hue fallback -> nothing to stash (matches the
    # un-hued scatter: a single-color plot has no legend entry).
    if hue is not None and palette and flags["hue"]:
        hue_label = legend_kws.pop("hue_label", hue)
        labels = list(palette.keys())
        colors = [palette[v] for v in labels]
        handles = create_legend_handles(
            labels=[str(v) for v in labels],
            colors=colors,
            edgecolors=[edgecolor] * len(labels) if edgecolor else None,
            alpha=alpha,
            linewidth=linewidth,
            markers=[marker if marker is not None else "o"] * len(labels),
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=hue_label,
                kind="hue",
                handles=handles,
                labels=[str(v) for v in labels],
            ),
        )

    render_entries(ax, flags=flags, legend_kws=legend_kws)
