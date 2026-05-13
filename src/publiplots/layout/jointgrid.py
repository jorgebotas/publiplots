"""
Bivariate + marginal-distribution composition (``pp.JointGrid`` / ``pp.jointplot``).

JointGrid is a 3-axes composition built on top of :func:`publiplots.subplots`'s
asymmetric-grid support: a large main panel, a thin top marginal, a thin right
marginal, and an empty top-right corner. Any ``pp.*`` plot function can be
forwarded into the joint or marginal slots via :meth:`JointGrid.plot_joint` /
:meth:`JointGrid.plot_marginals`. The convenience wrapper
:func:`jointplot` maps a ``kind=`` string to a pair of plot functions.
"""

from typing import Callable, Optional, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


_KIND_REGISTRY: dict = {}
"""``kind → (joint_fn, marginal_fn)`` registry, populated by :func:`jointplot`.

Populated lazily on first call to avoid a circular import with
``publiplots.plot.*``. Keys are the string aliases accepted by
``pp.jointplot(kind=...)``.
"""


class JointGrid:
    """Three-axes composition: main bivariate panel + marginals on top and right.

    The grid is a 2×2 :func:`publiplots.subplots` layout with the top-right
    corner hidden. The main panel shares its x-axis with the top marginal and
    its y-axis with the right marginal — so zooming the main panel
    auto-scrolls the matching marginal.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-form frame with the ``x`` and ``y`` variables.
    x, y : str
        Column names for the bivariate plot.
    height : float, default 80
        Total grid budget, in **millimeters**, as a square. Main cell
        becomes ``height * ratio/(ratio+1)`` on each side; marginals are
        ``height / (ratio+1)`` thick. Matches seaborn's ``JointGrid(height=)``
        kwarg modulo the unit (mm vs inches).
    ratio : int, default 5
        Marginal-to-main ratio. With the default, the main cell is 5/6 of
        the grid in each direction and the marginals are 1/6. Matches
        :class:`seaborn.JointGrid`'s default.
    space : float, optional
        Gap between panels, in **millimeters**. Applied symmetrically to
        ``wspace`` and ``hspace`` on the underlying ``pp.subplots`` call.
        When ``None`` (default), scales with ``height`` as
        ``height * 0.025`` — 2 mm at the default 80 mm grid, 1 mm at 40
        mm, 5 mm at 200 mm. Pass an explicit value (e.g. ``space=2``) to
        lock the gap in mm and preserve cross-figure consistency when
        composing multiple JointGrids of different sizes.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The underlying figure.
    ax_joint : matplotlib.axes.Axes
        The main bivariate panel (bottom-left cell).
    ax_marg_x : matplotlib.axes.Axes
        The top marginal (shares x with ``ax_joint``).
    ax_marg_y : matplotlib.axes.Axes
        The right marginal (shares y with ``ax_joint``).
    data : pandas.DataFrame
        Reference to the input data.
    x, y : str
        The column names passed to the constructor.

    Notes
    -----
    **Joint-panel legend routing.** ``__init__`` pre-attaches a
    right-side legend band anchored on ``ax_marg_y`` via
    ``pp.legend(anchor=self.ax_marg_y, side="right")``. Anything
    stashed on ``ax_joint`` — most commonly the continuous-hue
    colorbar from :func:`publiplots.hexbinplot` — is claimed by that
    band and rendered past the right marginal, top-aligned with the
    grid edge (the publiplots default for ``side="right"``). Without
    this routing, :class:`SubplotsAutoLayout` would measure the
    colorbar on ``ax_joint`` and grow the main-panel column's
    ``right`` reservation, pushing the right marginal away from the
    joint panel.

    The pre-attached band has zero layout cost when nothing is stashed
    (e.g., a plain :func:`publiplots.scatterplot` joint) — ``pp.legend``
    renders nothing and the right-marginal column's ``right`` reservation
    stays 0 mm.

    To customize legend placement (e.g., put the colorbar on the
    bottom instead), override after construction with another
    ``pp.legend`` call scoped via ``collect=[...]`` to claim the entry
    away from the default band.

    Examples
    --------
    Compose arbitrary plot types:

    >>> g = pp.JointGrid(data=df, x="umap1", y="umap2")
    >>> g.plot_joint(pp.hexbinplot, gridsize=20)
    >>> g.plot_marginals(pp.histplot, bins=40)

    Use the convenience wrapper for canonical kinds:

    >>> pp.jointplot(data=df, x="umap1", y="umap2", kind="hex")
    """

    _SPACE_AUTOSCALE = 0.025  # default mm-gap as a fraction of `height`

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        x: str,
        y: str,
        height: float = 80.0,
        ratio: int = 5,
        space: Optional[float] = None,
    ):
        from publiplots.layout.subplots import subplots as _pp_subplots

        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")

        # `space` stays in mm — publiplots' canonical unit for inter-panel
        # gaps. The default scales with `height` so the gap reads well
        # across a wide range of grid sizes (2 mm at 80 mm, 1 mm at 40 mm,
        # 5 mm at 200 mm). An explicit value is used as-is, preserving
        # cross-figure consistency for users who lock the gap in mm.
        if space is None:
            space = height * self._SPACE_AUTOSCALE
        elif space < 0:
            raise ValueError(f"space must be non-negative, got {space}")

        per_cell = height / 2.0  # axes_size is per-cell; grid budget is height
        # Lock the inside-facing reservations (the ones that face the joint
        # panel) to 0 mm. Without this, ``SubplotsAutoLayout`` measures
        # ~1.5 mm of baseline tightbbox padding off the empty marginal-edge
        # axes (even with ticks + labels hidden), which inflates
        # ``xlabel_space[0]`` (top-marginal bottom edge) and
        # ``ylabel_space[1]`` (right-marginal left edge) asymmetrically and
        # breaks the visual symmetry of the joint↔marginal gaps. The outer
        # edges (``xlabel_space[1]`` and ``ylabel_space[0]``) stay
        # auto-measured so the joint's own xlabel/ylabel decoration grows
        # the figure as usual.
        fig, axes = _pp_subplots(
            2, 2,
            axes_size=(per_cell, per_cell),
            width_ratios=[ratio, 1],
            height_ratios=[1, ratio],
            sharex="col",
            sharey="row",
            wspace=space,
            hspace=space,
            xlabel_space=(0.0, None),
            ylabel_space=(None, 0.0),
            title_space=(None, 0.0),
            right=(0.0, None),
        )

        axes[0, 1].set_visible(False)
        self.fig: Figure = fig
        self.ax_joint: Axes = axes[1, 0]
        self.ax_marg_x: Axes = axes[0, 0]
        self.ax_marg_y: Axes = axes[1, 1]
        self.data = data
        self.x = x
        self.y = y

        self._style_marginals()
        self._attach_legend_band()

    def _attach_legend_band(self) -> None:
        """Pre-attach a right-side legend band anchored on the right marginal.

        Any colorbar stashed on ``ax_joint`` (e.g. by ``pp.hexbinplot``) is
        claimed by this band and rendered past the right marginal. The band
        absorbs ``ax_marg_y``'s ``right`` reservation (the empty col-1
        outside edge), not ``ax_joint``'s — so the joint ↔ right-marginal
        gap stays tight regardless of the colorbar.

        When there are no stashed entries (e.g. plain ``pp.scatterplot``),
        the band renders nothing and the ``right`` reservation stays 0 mm
        — zero layout cost.
        """
        from publiplots.utils.legend_group import legend as _pp_legend
        _pp_legend(anchor=self.ax_marg_y, side="right")

    def _style_marginals(self) -> None:
        """Hide tick marks and labels on the marginal edges that face the joint.

        The top marginal's bottom edge shares x with the joint via
        ``sharex='col'``, and the right marginal's left edge shares y via
        ``sharey='row'`` — tick labels on those inside-facing edges would
        duplicate the joint's own axis decoration. Tick marks themselves
        are also hidden so they don't visually cross the joint↔marginal gap.

        Even with ticks hidden, matplotlib's ``get_tightbbox`` still
        reports ~1.5 mm of inherent baseline padding for the empty axis.
        :meth:`__init__` neutralizes that by passing per-position lock
        tuples (``xlabel_space=(0.0, None)``, ``ylabel_space=(None, 0.0)``,
        ``title_space=(None, 0.0)``, ``right=(0.0, None)``) to
        ``pp.subplots``, pinning the four inside-facing reservations to
        0 mm so the joint↔top-marginal and joint↔right-marginal gaps stay
        symmetric and equal to ``space``.
        """
        self.ax_marg_x.tick_params(axis="x", bottom=False, labelbottom=False)
        self.ax_marg_x.set_xlabel("")
        self.ax_marg_y.tick_params(axis="y", left=False, labelleft=False)
        self.ax_marg_y.set_ylabel("")

    def plot_joint(self, fn: Callable, **kwargs) -> "JointGrid":
        """Draw ``fn`` on the main panel.

        Forwards ``data=self.data, x=self.x, y=self.y, ax=self.ax_joint``
        plus any extra ``**kwargs``. Works with any publiplots plot
        function that accepts those kwargs (scatterplot, hexbinplot,
        histplot, heatmap, etc.).

        Returns
        -------
        JointGrid
            ``self``, so calls can be chained.
        """
        fn(data=self.data, x=self.x, y=self.y, ax=self.ax_joint, **kwargs)
        return self

    def plot_marginals(self, fn: Callable, **kwargs) -> "JointGrid":
        """Draw ``fn`` twice — once for each marginal.

        Top marginal receives ``x=self.x``, right marginal receives
        ``y=self.y``. Plot functions that support swapping ``x=``/``y=``
        to flip orientation work directly: ``pp.histplot`` (the default
        marginal for :func:`jointplot`) and ``pp.stripplot``.
        ``pp.violinplot`` and ``pp.boxplot`` also support 1D usage
        (only ``x`` or only ``y``) and work as marginals out of the box.

        Returns
        -------
        JointGrid
            ``self``, so calls can be chained.
        """
        fn(data=self.data, x=self.x, ax=self.ax_marg_x, **kwargs)
        fn(data=self.data, y=self.y, ax=self.ax_marg_y, **kwargs)
        return self

    def plot(
        self,
        joint_fn: Callable,
        marginal_fn: Callable,
        **kwargs,
    ) -> "JointGrid":
        """Shortcut for ``plot_joint(joint_fn)`` + ``plot_marginals(marginal_fn)``.

        ``**kwargs`` is forwarded to both calls. For per-function kwargs,
        use ``plot_joint`` and ``plot_marginals`` separately.
        """
        self.plot_joint(joint_fn, **kwargs)
        self.plot_marginals(marginal_fn, **kwargs)
        return self


def jointplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    kind: str = "scatter",
    height: float = 80.0,
    ratio: int = 5,
    space: Optional[float] = None,
    **kwargs,
) -> JointGrid:
    """Convenience wrapper: build a :class:`JointGrid` and plot in one call.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-form frame.
    x, y : str
        Column names for the bivariate plot.
    kind : {"scatter", "hex", "hist", "kde", "reg", "resid"}, default "scatter"
        Which joint + marginal plot functions to draw:

        - ``"scatter"`` — :func:`publiplots.scatterplot` joint +
          :func:`publiplots.histplot` marginals.
        - ``"hex"`` — :func:`publiplots.hexbinplot` joint (with colorbar
          routed to a figure-level band) + :func:`publiplots.histplot`
          marginals.
        - ``"hist"`` — :func:`publiplots.histplot` 2D heatmap joint +
          :func:`publiplots.histplot` 1D histogram marginals. A
          discretized alternative to ``"hex"``.
        - ``"kde"`` — :func:`publiplots.kdeplot` 2D contour joint +
          :func:`publiplots.kdeplot` 1D density marginals.
        - ``"reg"`` — :func:`publiplots.regplot` joint +
          :func:`publiplots.histplot` marginals (regplot has no 1D mode).
        - ``"resid"`` — :func:`publiplots.residplot` joint +
          :func:`publiplots.histplot` marginals.
    height : float, default 80
        Total grid budget in mm (square).
    ratio : int, default 5
        Marginal-to-main ratio.
    space : float, optional
        Panel gap in mm. ``None`` (default) auto-scales with ``height``
        — see :class:`JointGrid` for the formula.
    **kwargs
        Forwarded to both the joint and marginal plot calls. For per-slot
        kwargs, construct a :class:`JointGrid` and call ``plot_joint`` /
        ``plot_marginals`` separately.

    Returns
    -------
    JointGrid
        The constructed grid, with both joint and marginal panels plotted.
    """
    _ensure_kind_registry()
    if kind not in _KIND_REGISTRY:
        raise ValueError(
            f"kind={kind!r} not supported. Available kinds: "
            f"{sorted(_KIND_REGISTRY)}."
        )
    joint_fn, marginal_fn = _KIND_REGISTRY[kind]
    grid = JointGrid(data=data, x=x, y=y, height=height, ratio=ratio, space=space)
    grid.plot(joint_fn, marginal_fn, **kwargs)
    return grid


def _ensure_kind_registry() -> None:
    """Populate ``_KIND_REGISTRY`` on first use.

    Kept lazy so importing ``publiplots.layout.jointgrid`` doesn't pull in
    the full plot module graph at import time.
    """
    if _KIND_REGISTRY:
        return
    from publiplots.plot.hexbin import hexbinplot
    from publiplots.plot.hist import histplot
    from publiplots.plot.kdeplot import kdeplot
    from publiplots.plot.regplot import regplot
    from publiplots.plot.residplot import residplot
    from publiplots.plot.scatter import scatterplot
    _KIND_REGISTRY.update({
        "scatter": (scatterplot,  histplot),
        "hex":     (hexbinplot,   histplot),
        "hist":    (histplot,     histplot),
        "kde":     (kdeplot,      kdeplot),
        "reg":     (regplot,      histplot),
        "resid":   (residplot,    histplot),
    })
