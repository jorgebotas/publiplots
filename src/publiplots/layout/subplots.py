"""
Public API: pp.subplots() — fixed-axes, flexible-canvas subplot factory.

publiplots' flagship 0.10 layout API. Declared axes dimensions are in
**millimeters** and are inviolate; the figure canvas grows to
accommodate auto-measured decorations (titles, labels, tick labels) on
the first draw. This is the opposite mental model from
:func:`matplotlib.pyplot.subplots`, which fixes the figure and lets
axes shrink to fit.

Users coming from seaborn / matplotlib should note:

- ``axes_size=(w_mm, h_mm)`` is in **mm**, not inches.
- ``figsize=`` is not accepted — publiplots owns the figure geometry.
  See :func:`reject_figsize` for the rationale.

Follow-up PR(s) will add a Composer for cross-figure page layout.
"""

import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from publiplots.themes.rcparams import resolve_param
from publiplots.layout.figure_layout import FigureLayout
from publiplots.layout.auto_layout import SubplotsAutoLayout


_MM2INCH = 1 / 25.4
_RESERVATION_KEYS = (
    "title_space", "xlabel_space", "ylabel_space", "right",
    "hspace", "wspace", "outer_pad",
)
_LAYOUT_ENGINE_KWARGS = ("layout", "constrained_layout", "tight_layout")


def reject_figsize(kwargs: dict) -> None:
    """Raise :class:`TypeError` if ``figsize`` appears in ``kwargs``.

    publiplots plot functions no longer accept ``figsize=`` (removed in
    0.10). The figure canvas is owned by :func:`subplots`, which sizes
    it from ``axes_size`` (mm) plus auto-measured decoration
    reservations. To customize axes dimensions, compose with
    :func:`subplots` and pass ``ax=``.

    Parameters
    ----------
    kwargs : dict
        The ``**kwargs`` dict passed into a plot function. Mutated only
        by raising; not modified otherwise.

    Raises
    ------
    TypeError
        If ``"figsize"`` is a key in ``kwargs``.

    Examples
    --------
    >>> import publiplots as pp
    >>> fig, ax = pp.subplots(axes_size=(80, 50))  # 80 mm x 50 mm
    >>> pp.barplot(data=df, x='x', y='y', ax=ax)
    """
    if "figsize" in kwargs:
        raise TypeError(
            "publiplots plot functions no longer accept figsize=. "
            "Use pp.subplots(axes_size=(w_mm, h_mm)) to customize axes "
            "dimensions, then pass ax= to the plot function."
        )


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    axes_size: Union[Tuple[float, float], float, None] = None,
    sharex: Union[bool, str] = False,
    sharey: Union[bool, str] = False,
    title_space: Optional[float] = None,
    xlabel_space: Optional[float] = None,
    ylabel_space: Optional[float] = None,
    right: Optional[float] = None,
    hspace: Optional[float] = None,
    wspace: Optional[float] = None,
    outer_pad: Optional[float] = None,
    **fig_kw,
):
    """
    Create a figure and a grid of axes with deterministic axes dimensions.

    publiplots' flagship 0.10 layout API. Every axes in the grid has
    exactly ``axes_size`` **millimeters** as its spine bounding box —
    this is the inviolate quantity. The figure canvas is sized to fit
    the grid plus auto-measured decorations (titles, axis labels, tick
    labels), which are remeasured on first draw. Any per-side
    reservation passed explicitly is locked and never remeasured.

    This is the opposite mental model from
    :func:`matplotlib.pyplot.subplots`, which fixes the figure and lets
    axes shrink. When comparing to matplotlib/seaborn code, note that
    ``axes_size`` is in **mm** (not inches) and ``figsize=`` is
    rejected — publiplots owns the figure geometry. See
    :func:`reject_figsize` for the rationale.

    Parameters
    ----------
    nrows, ncols : int, default 1
        Grid shape (must be >= 1).
    axes_size : (float, float) or float, in mm, optional
        Declared axes bounding box, in **millimeters**. A scalar is
        coerced to ``(s, s)``. If ``None``, falls back to
        ``pp.rcParams["subplots.axes_size"]`` — the baseline publication
        default is ``(70, 50)`` mm.
    sharex, sharey : bool or {"all", "row", "col", "none"}, default False
        Axis-sharing semantics, matching :func:`matplotlib.pyplot.subplots`.
    title_space, xlabel_space, ylabel_space, right : float, optional
        Per-cell reservations, in millimeters. ``None`` means: take the
        initial value from ``pp.rcParams["subplots.<name>"]``, then
        auto-measure on first draw. Passing a float locks the value and
        disables auto-measurement for that side.
    hspace, wspace, outer_pad : float, optional
        Gaps between rows/cols and outer margin, in millimeters.
        ``None`` falls back to ``pp.rcParams["subplots.<name>"]``. These
        are never auto-measured.
    **fig_kw
        Forwarded to :func:`matplotlib.pyplot.figure`. ``figsize=`` is
        rejected (see :func:`reject_figsize`); matplotlib layout-engine
        kwargs (``layout``, ``constrained_layout``, ``tight_layout``)
        are ignored with a :class:`UserWarning`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : matplotlib.axes.Axes or numpy.ndarray of Axes
        Shape matches :func:`matplotlib.pyplot.subplots` with
        ``squeeze=True``.

    Notes
    -----
    **Auto-growing canvas.** The figure starts at the size computed from
    ``axes_size`` plus initial rcParams reservations. On first draw,
    decorations (titles, axis labels, tick labels) are measured and the
    reservations for unlocked sides are expanded to fit. The axes bbox
    itself never changes — only the canvas around it grows.

    **Legend overhangs.** If you attach a :func:`publiplots.legend` band
    with ``side="right"``/``"bottom"``/``"left"``/``"top"``, the extra
    space for the legend is auto-computed on first draw from the
    rendered group's width/height. There is no ``legend_column`` kwarg
    to set by hand.

    Examples
    --------
    Single axes (defaults):

    >>> import publiplots as pp
    >>> fig, ax = pp.subplots()
    >>> pp.barplot(data=df, x='category', y='value', ax=ax)

    Explicit mm axes size:

    >>> fig, ax = pp.subplots(axes_size=(80, 50))  # 80 mm x 50 mm

    Grid of axes with row-shared y-axis:

    >>> fig, axes = pp.subplots(2, 3, axes_size=(40, 30), sharey='row')

    Locked title reservation (no auto-measurement for this side):

    >>> fig, ax = pp.subplots(axes_size=(60, 40), title_space=8)

    See Also
    --------
    reject_figsize : Guard used by plot functions to reject ``figsize=``.
    publiplots.rcParams : Where the ``subplots.*`` defaults live.
    """
    # --- validation ---------------------------------------------------------
    if nrows < 1:
        raise ValueError(f"nrows must be >= 1, got {nrows}")
    if ncols < 1:
        raise ValueError(f"ncols must be >= 1, got {ncols}")

    if axes_size is None:
        axes_size = resolve_param("subplots.axes_size", None)
    if isinstance(axes_size, (int, float)):
        if axes_size <= 0:
            raise ValueError(f"axes_size must be positive, got {axes_size}")
        axes_size_t: Tuple[float, float] = (float(axes_size), float(axes_size))
    else:
        try:
            w_ax, h_ax = axes_size
        except (TypeError, ValueError):
            raise ValueError(
                f"axes_size must be a positive scalar or (width, height) tuple, got {axes_size!r}"
            )
        if w_ax <= 0 or h_ax <= 0:
            raise ValueError(f"axes_size must be positive, got {axes_size}")
        axes_size_t = (float(w_ax), float(h_ax))

    if "figsize" in fig_kw:
        raise TypeError(
            "pp.subplots() does not accept figsize; use axes_size (mm). "
            "Figure size is computed from axes_size + reservations."
        )
    if "legend_column" in fig_kw:
        raise TypeError(
            "pp.subplots() no longer accepts legend_column. Attach a "
            "pp.legend(side=...) band to the figure; the column is "
            "auto-sized based on the rendered group width."
        )
    for k in _LAYOUT_ENGINE_KWARGS:
        if k in fig_kw:
            warnings.warn(
                f"publiplots manages layout; ignoring {k}={fig_kw[k]!r}",
                UserWarning,
                stacklevel=2,
            )
            fig_kw.pop(k)

    # --- resolve reservation defaults & track locked sides ----------------
    _ROW_SIDES = ("title_space", "xlabel_space")
    _COL_SIDES = ("ylabel_space", "right")
    _SCALAR_SIDES = ("hspace", "wspace", "outer_pad")

    user_values = dict(
        title_space=title_space, xlabel_space=xlabel_space,
        ylabel_space=ylabel_space, right=right,
        hspace=hspace, wspace=wspace, outer_pad=outer_pad,
    )
    locked = {k for k, v in user_values.items() if v is not None}

    resolved = {}
    for side in _ROW_SIDES + _COL_SIDES:
        user_val = user_values[side]
        length = nrows if side in _ROW_SIDES else ncols
        if user_val is None:
            default_scalar = resolve_param(f"subplots.{side}", None)
            if default_scalar < 0:
                raise ValueError(f"{side} default must be non-negative, got {default_scalar}")
            resolved[side] = (float(default_scalar),) * length
        elif isinstance(user_val, (int, float)) and not isinstance(user_val, bool):
            if user_val < 0:
                raise ValueError(f"{side} must be non-negative, got {user_val}")
            resolved[side] = (float(user_val),) * length
        else:
            try:
                tup = tuple(float(x) for x in user_val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{side} must be a scalar or sequence of numbers, got {user_val!r}"
                )
            if len(tup) != length:
                raise ValueError(
                    f"{side} must have length {length} for nrows={nrows} ncols={ncols}, "
                    f"got length {len(tup)}"
                )
            for i, v in enumerate(tup):
                if v < 0:
                    raise ValueError(f"{side}[{i}] must be non-negative, got {v}")
            resolved[side] = tup

    for side in _SCALAR_SIDES:
        val = resolve_param(f"subplots.{side}", user_values[side])
        if val < 0:
            raise ValueError(f"{side} must be non-negative, got {val}")
        resolved[side] = float(val)

    # --- build layout & figure --------------------------------------------
    layout = FigureLayout(
        nrows=nrows, ncols=ncols,
        axes_size=axes_size_t,
        legend_column=0.0,
        suptitle_space=0.0,
        **resolved,
    )
    W, H = layout.figure_size()
    fig = plt.figure(figsize=(W * _MM2INCH, H * _MM2INCH), layout=None, **fig_kw)

    # --- create axes with sharing semantics -------------------------------
    axes_matrix = _build_axes(fig, layout, sharex, sharey)
    fig._publiplots_axes = axes_matrix

    # --- attach auto-layout hook (skipped internally if all sides locked) -
    # Only lock the auto-measurable sides; hspace/wspace/outer_pad are not
    # auto-measured regardless.
    auto_locked = locked & {"title_space", "xlabel_space", "ylabel_space", "right"}
    # If every auto-measurable side is user-locked, SubplotsAutoLayout
    # skips the draw-event connection (see auto_layout.py).
    fig._publiplots_auto_layout = SubplotsAutoLayout(fig, layout, locked=auto_locked)

    # --- squeeze & return --------------------------------------------------
    arr = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            arr[r, c] = axes_matrix[r][c]
    return fig, _squeeze(arr, nrows, ncols)


def _build_axes(fig, layout, sharex, sharey):
    """Create axes in row-major order with plt.subplots-style sharing."""
    matrix = [[None] * layout.ncols for _ in range(layout.nrows)]
    for r in range(layout.nrows):
        for c in range(layout.ncols):
            share_x = _resolve_shared(sharex, matrix, r, c, axis="x")
            share_y = _resolve_shared(sharey, matrix, r, c, axis="y")
            kwargs = {}
            if share_x is not None:
                kwargs["sharex"] = share_x
            if share_y is not None:
                kwargs["sharey"] = share_y
            ax = fig.add_axes(layout.axes_position(r, c), **kwargs)
            matrix[r][c] = ax
    return matrix


def _resolve_shared(share, matrix, r, c, axis):
    """Return the axes to share with, or None."""
    if share in (False, "none"):
        return None
    if r == 0 and c == 0:
        return None
    if share in (True, "all"):
        return matrix[0][0]
    if share == "row":
        return matrix[r][0] if c > 0 else None
    if share == "col":
        return matrix[0][c] if r > 0 else None
    raise ValueError(f"share{axis} must be bool or one of 'all'/'row'/'col'/'none', got {share!r}")


def _squeeze(arr, nrows, ncols):
    if nrows == 1 and ncols == 1:
        return arr[0, 0]
    if nrows == 1:
        return arr[0, :]
    if ncols == 1:
        return arr[:, 0]
    return arr