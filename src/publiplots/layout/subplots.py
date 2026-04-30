"""
Public API: pp.subplots() — fixed-axes, flexible-canvas subplot factory.

Declared axes dimensions (in mm) are inviolate; the figure grows to
accommodate auto-measured decorations. Follow-up PR(s) will add
legend-width awareness and a Composer for cross-figure page layout.
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


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    axes_size: Union[Tuple[float, float], float],
    sharex: Union[bool, str] = False,
    sharey: Union[bool, str] = False,
    title_space: Optional[float] = None,
    xlabel_space: Optional[float] = None,
    ylabel_space: Optional[float] = None,
    right: Optional[float] = None,
    hspace: Optional[float] = None,
    wspace: Optional[float] = None,
    outer_pad: Optional[float] = None,
    legend_column: float = 0.0,
    **fig_kw,
):
    """
    Create a figure and a grid of axes with deterministic axes dimensions.

    Every axes in the grid has exactly ``axes_size`` mm as its spine
    bounding box. The figure size is computed to accommodate decorations
    (titles, axis labels, tick labels) which are auto-measured on first
    draw. Any per-side reservation passed explicitly is locked and never
    remeasured.

    Parameters
    ----------
    nrows, ncols : int, default 1
        Grid shape (must be >= 1).
    axes_size : (float, float) or float, in mm
        Declared axes bbox. Scalar is coerced to ``(s, s)``.
    sharex, sharey : bool or {"all", "row", "col", "none"}
        Axis-sharing semantics, matching ``plt.subplots``.
    title_space, xlabel_space, ylabel_space, right : float, optional
        Per-cell reservations in mm. ``None`` means: initial value from
        rcParams, then auto-measured on first draw. A float locks the
        value.
    hspace, wspace, outer_pad : float, optional
        Gaps and outer margin in mm. ``None`` falls back to rcParams.
        Never auto-measured.
    legend_column : float, default 0
        Extra width reserved on the far right of the figure (outside the
        grid, applied once — not per-cell). Intended for a single unified
        ``pp.legend_group`` anchored to the rightmost axes. Never
        auto-measured — opt-in only. Future: legend-width awareness will
        compute this automatically from registered legend content.
    **fig_kw
        Forwarded to ``plt.figure``. ``figsize`` is rejected; layout-
        engine kwargs are ignored with a warning.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or numpy.ndarray of Axes
        Shape matches ``plt.subplots(squeeze=True)``.
    """
    # --- validation ---------------------------------------------------------
    if nrows < 1:
        raise ValueError(f"nrows must be >= 1, got {nrows}")
    if ncols < 1:
        raise ValueError(f"ncols must be >= 1, got {ncols}")

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
    for k in _LAYOUT_ENGINE_KWARGS:
        if k in fig_kw:
            warnings.warn(
                f"publiplots manages layout; ignoring {k}={fig_kw[k]!r}",
                UserWarning,
                stacklevel=2,
            )
            fig_kw.pop(k)

    # --- resolve reservation defaults & track locked sides ----------------
    user_values = dict(
        title_space=title_space, xlabel_space=xlabel_space,
        ylabel_space=ylabel_space, right=right,
        hspace=hspace, wspace=wspace, outer_pad=outer_pad,
    )
    locked = {k for k, v in user_values.items() if v is not None}
    resolved = {}
    for k, v in user_values.items():
        val = resolve_param(f"subplots.{k}", v)
        if val < 0:
            raise ValueError(f"{k} must be non-negative, got {val}")
        resolved[k] = float(val)

    if legend_column < 0:
        raise ValueError(f"legend_column must be non-negative, got {legend_column}")

    # --- build layout & figure --------------------------------------------
    layout = FigureLayout(
        nrows=nrows, ncols=ncols,
        axes_size=axes_size_t,
        legend_column=float(legend_column),
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