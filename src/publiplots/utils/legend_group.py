"""
Shared legend band across multiple subplots.

MultiAxesLegendGroup composes one unified legend band anchored to a
chosen side of the figure (or a single axes) even when individual
legends/colorbars are attached to other axes in the same figure. This
is the primary tool for complex subplot layouts.
"""

import warnings
from typing import List, Optional, Sequence, Tuple, Union

# PR 4: grid-scope kwargs accept either a single int or an inclusive (start, end)
# tuple of ints. Defined once to keep the resolver and factory signatures aligned.
_RowColSpec = Union[int, Tuple[int, int]]

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.transforms import Bbox

from publiplots.utils.legend import LegendBuilder
from publiplots.utils.legend_entries import (
    LegendEntry,
    get_entries,
    is_continuous_hue,
)


def _inside_loc_from_side_align(side: str, align: str) -> str:
    """Map publiplots ``side`` + ``align`` to a matplotlib ``loc=`` string.

    Used by :func:`legend` when ``inside=True`` — the legend is rendered
    inside the anchor's rectangle via matplotlib's native corner-based
    placement, so we need to translate the publiplots band-mode grammar
    (``side`` + ``align``) to a matplotlib ``loc`` string.

    Mapping (13 reachable positions):

    ===========  ===================  =====================  ===================
    side         align='start'        align='center'         align='end'
    ===========  ===================  =====================  ===================
    ``left``     ``upper left``       ``center left``        ``lower left``
    ``right``    ``upper right``      ``center right``       ``lower right``
    ``top``      ``upper left``       ``upper center``       ``upper right``
    ``bottom``   ``lower left``       ``lower center``       ``lower right``
    ``center``   ``center``           ``center``             ``center``
    ===========  ===================  =====================  ===================

    ``side='center'`` collapses every ``align`` to matplotlib's plain
    ``'center'`` — the geometry can't differentiate.

    Parameters
    ----------
    side : {'left', 'right', 'top', 'bottom', 'center'}
    align : {'start', 'center', 'end'}

    Returns
    -------
    str
        A valid matplotlib ``Legend(loc=...)`` value.
    """
    if side == "center":
        return "center"
    table = {
        "left":   {"start": "upper left",  "center": "center left",  "end": "lower left"},
        "right":  {"start": "upper right", "center": "center right", "end": "lower right"},
        "top":    {"start": "upper left",  "center": "upper center", "end": "upper right"},
        "bottom": {"start": "lower left",  "center": "lower center", "end": "lower right"},
    }
    if side not in table:
        raise ValueError(
            f"side must be 'left' | 'right' | 'top' | 'bottom' | 'center', "
            f"got {side!r}"
        )
    if align not in table[side]:
        raise ValueError(
            f"align must be 'start' | 'center' | 'end', got {align!r}"
        )
    return table[side][align]


def _handle_repr(handle) -> str:
    """Cheap fingerprint of a matplotlib handle's key visual props.

    Used by ``MultiAxesLegendGroup._merge_entries`` to detect when
    two axes stashed the same label with a visually different handle
    (mismatched color / marker / linewidth). Implementation mirrors
    ``legend_entries._hash_handles`` but for a single handle so we
    can compare label-by-label without rehashing the whole entry.
    """
    parts = [type(handle).__name__]
    for attr in ("get_facecolor", "get_marker", "get_markersize",
                 "get_linewidth"):
        fn = getattr(handle, attr, None)
        if fn is not None:
            try:
                parts.append(repr(fn()))
            except Exception:
                pass
    return "|".join(parts)


def _resolve_grid_scope(
    fig: Figure,
    *,
    rows: Optional[_RowColSpec] = None,
    cols: Optional[_RowColSpec] = None,
    span: Optional[str] = None,
    ax: Optional[Sequence[Axes]] = None,
) -> Optional[List[Axes]]:
    """Translate grid-scope kwargs into a concrete axes list.

    Returns
    -------
    list of Axes
        The resolved scope when ``rows``/``cols``/``ax`` produce a
        concrete subset.
    None
        When ``rows``/``cols``/``span``/``ax`` are all None, OR when
        ``span='fig'`` (both mean "no grid scoping; fall through to
        figure-level"). Caller's responsibility to dispatch.

    Raises
    ------
    ValueError
        On out-of-range indices, missing ``_publiplots_axes`` matrix
        when one is required, conflicting kwargs, empty ``ax=`` list,
        or invalid ``span`` value.
    """
    # Normalize: count how many of the four are actually set.
    rows_set = rows is not None
    cols_set = cols is not None
    span_set = span is not None
    ax_set = ax is not None

    # All-None → fall through.
    if not (rows_set or cols_set or span_set or ax_set):
        return None

    # `ax=` is mutually exclusive with all three others.
    if ax_set and (rows_set or cols_set or span_set):
        collided = [n for n, s in (("rows", rows_set), ("cols", cols_set),
                                   ("span", span_set)) if s]
        raise ValueError(
            f"pp.legend: `ax=` and `{'`, `'.join(collided)}=` are mutually "
            "exclusive — they're alternative addressing modes. Pick one."
        )

    # `span=` is mutually exclusive with explicit rows/cols.
    if span_set and (rows_set or cols_set):
        collided = [n for n, s in (("rows", rows_set), ("cols", cols_set)) if s]
        raise ValueError(
            f"pp.legend: `span=` and `{'`, `'.join(collided)}=` are mutually "
            "exclusive. Use `span` for sugar OR `rows`/`cols` for explicit "
            "scoping."
        )

    # Explicit ax= path — no _publiplots_axes lookup needed.
    if ax_set:
        ax_list = list(ax)
        if not ax_list:
            raise ValueError(
                "pp.legend: `ax=` was an empty sequence. Pass at least one "
                "Axes, or omit `ax=` for figure-level scoping."
            )
        return ax_list

    # `span=` path.
    if span_set:
        if span not in ("row", "col", "fig"):
            raise ValueError(
                f"pp.legend: span={span!r} invalid. Expected 'row', 'col', "
                "or 'fig'."
            )
        if span == "fig":
            return None  # full figure → caller falls through
        # 'row'/'col' need a positional anchor; the caller (`legend()`)
        # is responsible for passing that context. The resolver itself
        # raises here because we have no anchor.
        raise ValueError(
            f"pp.legend: span={span!r} requires a positional Axes anchor "
            "(`pp.legend(ax_anchor, span='row')`). Without an anchor, use "
            f"`rows=`/`cols=` for explicit indices or `span='fig'` for the "
            "full figure."
        )

    # `rows=`/`cols=` path — needs the _publiplots_axes matrix.
    # Note: pp.Canvas does NOT currently attach _publiplots_axes (PR 7
    # territory). The error message therefore points only at pp.subplots.
    matrix = getattr(fig, "_publiplots_axes", None)
    if not matrix:
        raise ValueError(
            "pp.legend: `rows=`/`cols=` requires a figure built by "
            "`pp.subplots` (no `_publiplots_axes` matrix on this figure). "
            "For raw matplotlib figures or pp.Canvas figures, use "
            "`ax=[ax1, ax2, ...]` instead."
        )

    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0

    def _is_int_like(v: object) -> bool:
        # Accept Python int + numpy integer; reject bool, float, str, etc.
        # Bool is a subclass of int in Python — reject explicitly.
        if isinstance(v, bool):
            return False
        if isinstance(v, int):
            return True
        try:
            import numpy as _np
            return isinstance(v, _np.integer)
        except ImportError:
            return False

    def _normalize_range(name: str, value: object, length: int) -> tuple[int, int]:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    f"pp.legend: `{name}=` tuple must be (start, end) — got "
                    f"{value!r}."
                )
            start, end = value
            if not (_is_int_like(start) and _is_int_like(end)):
                raise ValueError(
                    f"pp.legend: `{name}={value!r}` — tuple elements must be "
                    f"integers (got types {type(start).__name__}, "
                    f"{type(end).__name__})."
                )
            start = int(start)
            end = int(end)
        elif _is_int_like(value):
            start = end = int(value)
        else:
            raise ValueError(
                f"pp.legend: `{name}={value!r}` — must be int or "
                f"(start, end) tuple of ints (got {type(value).__name__})."
            )
        # Disallow negative indices: explicit, no Python wrap-around.
        if start < 0 or end < 0:
            last = length - 1
            raise ValueError(
                f"pp.legend: `{name}={value!r}` — negative indices are not "
                f"supported. Use `{name}={last}` for the last "
                f"{'row' if name == 'rows' else 'column'}."
            )
        # Disallow inverted ranges with a clearer message than "out of range".
        if start > end:
            raise ValueError(
                f"pp.legend: `{name}={value!r}` has start > end. Use "
                f"`{name}=({end}, {start})` to specify an inclusive range."
            )
        if not (start < length and end < length):
            raise ValueError(
                f"pp.legend: `{name}={value!r}` out of range for "
                f"_publiplots_axes shape ({n_rows}, {n_cols})."
            )
        return start, end

    if rows_set:
        r_start, r_end = _normalize_range("rows", rows, n_rows)
    else:
        r_start, r_end = 0, n_rows - 1

    if cols_set:
        c_start, c_end = _normalize_range("cols", cols, n_cols)
    else:
        c_start, c_end = 0, n_cols - 1

    out: List[Axes] = []
    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            out.append(matrix[r][c])
    return out


def _expand_span_with_anchor(
    fig: Figure,
    anchor: Axes,
    span: str,
) -> List[Axes]:
    """Expand `span='row'/'col'` against a positional Axes anchor.

    Locates ``anchor`` in ``fig._publiplots_axes`` and returns the full
    row or column containing it.

    Raises
    ------
    ValueError
        If the figure has no ``_publiplots_axes`` matrix, or if
        ``anchor`` is not in that matrix.
    """
    matrix = getattr(fig, "_publiplots_axes", None)
    if not matrix:
        raise ValueError(
            "pp.legend: `span='row'`/`'col'` requires a figure built by "
            "`pp.subplots` or `pp.Canvas` (no `_publiplots_axes` matrix on "
            "this figure)."
        )
    target_id = id(anchor)
    for r, row in enumerate(matrix):
        for c, ax_in_row in enumerate(row):
            if id(ax_in_row) == target_id:
                if span == "row":
                    return list(row)
                if span == "col":
                    return [matrix[rr][c] for rr in range(len(matrix))]
                raise ValueError(
                    f"pp.legend: span={span!r} not 'row' or 'col'."
                )
    raise ValueError(
        "pp.legend: positional anchor was not found in this figure's "
        "`_publiplots_axes` matrix."
    )


class _ScopeAnchor:
    """Decoration-agnostic anchor over a scope of axes.

    Duck-types the small slice of ``matplotlib.axes.Axes`` that
    ``LegendBuilder`` and ``LayoutReactor`` use (``get_position()`` +
    ``get_figure()`` + ``get_window_extent()``). Designed to become the
    single master anchor class for ``pp.legend`` — handling single-axes,
    sub-region (row/col), and full-grid scopes uniformly.

    Key design invariant: ``get_position()`` returns the **bare**
    bounding rect of the scope's axes (union of ``ax.get_position()``).
    Decoration-awareness (tick labels, axis labels, titles, legend
    bands) is applied one layer up by
    ``SubplotsAutoLayout._measure_one_group``. This keeps the anchor
    simple and testable while leaving geometric composition where the
    rest of the layout pipeline already lives.

    - Single axes → degenerate case: ``get_position()`` returns the
      axes rect.
    - Multi axes (incl. full grid) → bounding rect of the scope.

    For commit 1 of the v0.10.0 legend unification this class is
    introduced behind ``_GridAnchor``: ``_GridAnchor.__init__``
    constructs a ``_ScopeAnchor`` for reference, but continues to use
    its own decoration-aware ``get_position()`` logic for back-compat
    until later commits route ``MultiAxesLegendGroup`` through
    ``_ScopeAnchor`` directly.
    """

    def __init__(self, axes_list: List[Axes], fig: Figure) -> None:
        self._axes_list: List[Axes] = list(axes_list)
        self._fig = fig

    def get_figure(self) -> Figure:
        return self._fig

    def get_position(self) -> Bbox:
        """Bare union rect of the scope's axes, in figure fractions.

        No decoration awareness — callers that need decoration-aware
        placement must apply offsets externally (that logic lives in
        ``SubplotsAutoLayout._measure_one_group``).
        """
        if not self._axes_list:
            return Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
        x0 = y0 = 1.0
        x1 = y1 = 0.0
        for ax in self._axes_list:
            pos = ax.get_position()
            x0 = min(x0, pos.x0)
            y0 = min(y0, pos.y0)
            x1 = max(x1, pos.x1)
            y1 = max(y1, pos.y1)
        if x0 >= x1 or y0 >= y1:
            return Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
        return Bbox.from_extents(x0, y0, x1, y1)

    def get_window_extent(self, renderer=None) -> Bbox:
        """Pixel-space bbox of ``get_position()`` against the figure."""
        pos = self.get_position()
        fig_w = self._fig.get_window_extent().width
        fig_h = self._fig.get_window_extent().height
        return Bbox.from_extents(
            pos.x0 * fig_w, pos.y0 * fig_h,
            pos.x1 * fig_w, pos.y1 * fig_h,
        )

    @property
    def is_single(self) -> bool:
        """True iff the scope is a single axes (degenerate case)."""
        return len(self._axes_list) == 1

    @property
    def is_full_grid(self) -> bool:
        """True iff the scope covers every axis in ``fig._publiplots_axes``.

        Falls back to ``len(axes_list) == len(fig.axes)`` when the
        figure was not constructed by publiplots and therefore carries
        no ``_publiplots_axes`` matrix.
        """
        matrix = getattr(self._fig, "_publiplots_axes", None)
        if matrix is None:
            return len(self._axes_list) == len(self._fig.axes)
        grid_ids = {id(ax) for row in matrix for ax in row}
        if not grid_ids:
            return len(self._axes_list) == len(self._fig.axes)
        scope_ids = {id(ax) for ax in self._axes_list}
        return scope_ids == grid_ids

    def scope_axes(self) -> List[Axes]:
        """Return the underlying list of axes in the scope."""
        return self._axes_list


class _GridAnchor:
    """Virtual Axes proxy spanning the whole publiplots subplot grid
    **including its decorations** (tick labels, axis labels, titles).

    Duck-types the small slice of ``matplotlib.axes.Axes`` that
    ``LegendBuilder`` and ``LayoutReactor`` use (``get_position()`` +
    ``get_figure()`` + ``get_window_extent()``). Lets figure-anchored
    ``pp.legend_group`` share exactly the same placement machinery as
    axes-anchored groups — the anchor just happens to be the decorated
    grid rectangle (the outer boundary of everything *except* legend
    bands) rather than a single axes' rectangle.

    The decorated-grid bbox is computed from the ``FigureLayout``:

        x0 = (outer_pad + legend_band_left)     / W
        x1 = (W - outer_pad - legend_column)    / W
        y0 = (outer_pad + legend_band_bottom)   / H
        y1 = (H - outer_pad - legend_band_top)  / H

    Anchoring here (rather than the axes-rectangle union) guarantees
    that side='bottom' sits BELOW all xlabel_space, side='top' sits
    ABOVE all title_space, side='left' sits LEFT of all ylabel_space,
    etc. — i.e., the legend doesn't overlap with axis decorations.

    .. deprecated::
        Scheduled for removal in v0.10.0. Use :class:`_ScopeAnchor`
        instead; decoration-awareness will move into
        ``SubplotsAutoLayout._measure_one_group`` so the anchor itself
        can stay decoration-agnostic.
    """

    def __init__(self, fig: Figure) -> None:
        self._fig = fig
        # Delegate to _ScopeAnchor(full_grid_flat, fig) for the actual
        # geometry. Kept as a separate class for now because its
        # get_position() ALSO respects decorations (reads the
        # FigureLayout directly), which _ScopeAnchor deliberately does
        # NOT — decoration-awareness lives one layer up in
        # SubplotsAutoLayout._measure_one_group. The shim preserves
        # back-compat until commit 2 routes MultiAxesLegendGroup
        # through _ScopeAnchor directly.
        matrix = getattr(fig, "_publiplots_axes", None)
        if matrix:
            flat = [ax for row in matrix for ax in row]
        else:
            flat = list(fig.axes)
        self._scope_anchor: Optional[_ScopeAnchor] = (
            _ScopeAnchor(flat, fig) if flat else None
        )

    def get_figure(self) -> Figure:
        return self._fig

    def get_position(self) -> Bbox:
        """Decorated-grid bbox in figure fractions."""
        layout = getattr(self._fig, "_publiplots_layout", None)
        if layout is not None:
            W, H = layout.figure_size()
            # Sanity guard: avoid division by zero if the layout hasn't
            # measured yet (e.g., during the first draw before sizes are
            # known).
            if W > 0 and H > 0:
                x0 = (layout.outer_pad + layout.legend_band_left) / W
                x1 = (W - layout.outer_pad - layout.legend_column) / W
                y0 = (layout.outer_pad + layout.legend_band_bottom) / H
                y1 = (
                    H
                    - layout.outer_pad
                    - layout.suptitle_space
                    - layout.legend_band_top
                ) / H
                return Bbox.from_extents(x0, y0, x1, y1)
        # Fallback when no publiplots layout is installed (figure built
        # by raw matplotlib): union the axes rectangles only.
        matrix = getattr(self._fig, "_publiplots_axes", None)
        if matrix is None:
            return Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
        x0 = y0 = 1.0
        x1 = y1 = 0.0
        for row in matrix:
            for ax in row:
                pos = ax.get_position()
                x0, y0 = min(x0, pos.x0), min(y0, pos.y0)
                x1, y1 = max(x1, pos.x1), max(y1, pos.y1)
        if x0 >= x1 or y0 >= y1:
            return Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
        return Bbox.from_extents(x0, y0, x1, y1)

    def get_window_extent(self, renderer=None):
        """Pixel extent of the decorated-grid bbox."""
        pos = self.get_position()
        fig_w = self._fig.get_window_extent().width
        fig_h = self._fig.get_window_extent().height
        return Bbox.from_extents(
            pos.x0 * fig_w, pos.y0 * fig_h,
            pos.x1 * fig_w, pos.y1 * fig_h,
        )


class MultiAxesLegendGroup:
    """
    Unified legend band across multiple axes.

    All elements share a single mm-based layout anchored to the chosen
    side of ``anchor``. Each element can be attached to a different
    axes (via ``ax=`` on ``add_*`` calls) for hit-testing and picking;
    its POSITION is always computed against the anchor's chosen edge
    regardless of which axes owns the artist.

    The group can be attached **before or after** the plot calls.
    Before is marginally more efficient — each plot sees the group,
    skips its own per-axis legend render, and stashes the entry for
    the group to collect. After is seamless: the group walks every
    axes on construction and removes per-axis legend artists whose
    titles match entries it will claim, then renders the shared
    legend and lets ``SubplotsAutoLayout`` shrink the per-column /
    per-row reservations to match. Inside legends for entries the
    group does NOT claim (different kind, or excluded via ``collect=``)
    survive the eviction.

    Parameters
    ----------
    anchor : Axes, optional
        The axes whose chosen edge defines the origin for the shared
        band. When ``None`` (default), the group is **figure-anchored**:
        it spans the full subplot grid on the chosen side (e.g.,
        ``side='right'`` anchors to the rightmost cells' right edge and
        extends vertically across every row). Pass an explicit axes to
        pin the band to that single cell instead (axes-anchored).
    side : {'right', 'bottom', 'left', 'top'}, default 'right'
        Which edge of the anchor the band grows outward from. ``'right'``
        is the classic publiplots outside-right column.
    figure : Figure, optional
        Figure to attach a figure-anchored group to. Defaults to
        ``plt.gcf()``; only needed when no current figure exists.
    collect : list or tuple of str, optional
        Names of entries to auto-collect from across the grid's stashed
        ``LegendEntry`` objects. ``None`` (default) collects everything.
        A list filters and orders — e.g. ``collect=['treatment', 'dose']``
        renders only those two names, in that order.
    axes : Axes or sequence of Axes, optional
        Scope collection to a subset of the figure's axes. ``None``
        (default) walks the full subplot grid. When set, the group
        collects stashed entries only from those axes and evicts
        per-axis legends only from those axes — letting multiple groups
        on the same figure render independent bands (e.g. a
        ``side='top'`` band for the top row and a ``side='right'``
        band for the bottom row).
    x_offset, y_offset, gap, column_spacing, vpad, max_width
        Same meaning as :class:`LegendBuilder` — all in millimeters.
    """

    # Default outward gap (mm) between the anchor's edge and the
    # legend's near side. Per-side mapping so the defaults can be tuned
    # independently in the future (e.g., if tick labels start crowding
    # a specific side more than others).
    _DEFAULT_X_OFFSET_MM = {"right": 2, "left": 2, "bottom": 2, "top": 2}

    # side → default orientation. Horizontal makes sense on top/bottom
    # where there's plenty of figure width; vertical stays the default
    # for right/left where width is narrow.
    _DEFAULT_ORIENTATION = {
        "right": "vertical", "left": "vertical",
        "bottom": "horizontal", "top": "horizontal",
    }

    # side → default along-edge alignment. Horizontal bands center to
    # balance the figure; vertical bands start at the top (current
    # 'upper' behaviour).
    _DEFAULT_ALIGN = {
        "right": "start", "left": "start",
        "bottom": "center", "top": "center",
    }

    def __init__(
        self,
        anchor: Optional[Axes] = None,
        collect: Optional[Sequence[str]] = None,
        *,
        side: str = "right",
        figure: Optional[Figure] = None,
        orientation: str = "auto",
        align: str = "auto",
        axes: Optional[Sequence[Axes]] = None,
        x_offset: Optional[float] = None,
        y_offset: Optional[float] = None,
        gap: float = 2,
        column_spacing: float = 5,
        vpad: Optional[float] = None,
        max_width: Optional[float] = None,
        inside: bool = False,
        clear_anchor: bool = True,
    ):
        # side='center' is only valid in inside mode (where it maps to
        # matplotlib loc='center'); band mode has no center edge to pin to.
        valid_sides = ("right", "left", "bottom", "top")
        if inside:
            valid_sides = valid_sides + ("center",)
        if side not in valid_sides:
            raise ValueError(
                f"side must be one of {valid_sides!r}, got {side!r}"
            )
        if orientation not in ("auto", "vertical", "horizontal"):
            raise ValueError(
                f"orientation must be 'auto' | 'vertical' | 'horizontal', "
                f"got {orientation!r}"
            )
        if align not in ("auto", "start", "center", "end"):
            raise ValueError(
                f"align must be 'auto' | 'start' | 'center' | 'end', got {align!r}"
            )
        self._side = side
        self._inside = bool(inside)
        self._clear_anchor = bool(clear_anchor)
        # side='center' only exists in inside mode and has no band-mode
        # default for orientation/align/offset; default to vertical+center.
        if side == "center":
            default_orientation = "vertical"
            default_align = "center"
            default_x_offset = 0.0
        else:
            default_orientation = self._DEFAULT_ORIENTATION[side]
            default_align = self._DEFAULT_ALIGN[side]
            default_x_offset = self._DEFAULT_X_OFFSET_MM[side]
        self._orientation = (
            default_orientation if orientation == "auto" else orientation
        )
        self._align = default_align if align == "auto" else align
        if x_offset is None:
            x_offset = default_x_offset
        # In inside mode, side='center' silently coerces align to 'center'
        # because matplotlib's loc='center' has no notion of start/end.
        if self._inside and self._side == "center":
            self._align = "center"
        # Inside mode requires a real anchor axes — there's no rectangle
        # to render into otherwise. Validated again at the factory layer
        # for a friendlier error; this is the defense-in-depth check.
        if self._inside and not isinstance(anchor, Axes):
            raise ValueError(
                "inside=True requires anchor=<Axes>; got anchor="
                f"{anchor!r}"
            )

        # Decide anchor kind. When the caller passes an explicit axes we
        # pin the band to that one axes (and its per-cell reservation
        # tuple grows). Otherwise span the whole grid via _GridAnchor.
        # LegendBuilder resolves the default vpad from self._anchor_ax
        # (real Axes → 0; _GridAnchor → 5) so both anchor modes end up
        # visually aligned with the axes top.
        if anchor is None:
            fig = figure if figure is not None else plt.gcf()
            self.anchor = _GridAnchor(fig)
            self._anchor_kind = "figure"
        else:
            self.anchor = anchor
            self._anchor_kind = "axes"

        if collect is not None:
            if isinstance(collect, str) or not hasattr(collect, "__iter__"):
                raise TypeError(
                    "collect must be None or a list/tuple of names; "
                    "got a bare string. Wrap in a list: collect=['name']"
                )
            collect = list(collect)
        self._collect = collect

        # Normalize axes= into a list of real Axes or None (full grid).
        # A single Axes is accepted for convenience: axes=ax.
        if axes is None:
            self._scope_axes = None
        elif isinstance(axes, Axes):
            self._scope_axes = [axes]
        else:
            self._scope_axes = list(axes)
            for a in self._scope_axes:
                if not isinstance(a, Axes):
                    raise TypeError(
                        f"axes must be an Axes or a sequence of Axes; "
                        f"got {type(a).__name__}"
                    )

        # Commit 4: default to True to preserve pre-0.10
        # ``pp.legend_group(anchor=ax)`` semantics (external band absorbs
        # the axes' per-cell reservation). The unified ``pp.legend()``
        # factory mutates this to False post-construction for its
        # single-axes scope — keeping the class default backward-
        # compatible while still activating _measure_one_group's
        # commit-3 guard for per-axes legends created via ``pp.legend(ax)``.
        # Inside mode never overhangs the anchor's edge: the legend lives
        # entirely inside the anchor's rectangle, so the cell's per-side
        # reservation must NOT grow.
        self._external_to_axis = not self._inside

        # Construct a _ScopeAnchor mirror in every path. Commit 2 stores it
        # for reference only — current geometry still flows through
        # self.anchor (which is a _GridAnchor or direct Axes). Commit 4
        # swaps consumers to read from self._scope_anchor uniformly.
        if self._anchor_kind == "figure":
            fig = self.anchor.get_figure()  # _GridAnchor has get_figure()
            matrix = getattr(fig, "_publiplots_axes", None)
            flat = [a for row in matrix for a in row] if matrix else list(fig.axes)
            self._scope_anchor = _ScopeAnchor(flat, fig)
        else:
            scope_list = self._scope_axes if self._scope_axes is not None else [self.anchor]
            fig = self.anchor.get_figure()
            self._scope_anchor = _ScopeAnchor(scope_list, fig)

        self._materialized = False
        self._warned_mismatch = False
        self._align_connected = False
        self._aligning = False  # re-entrancy guard for _on_draw_align
        # Mm the band overhangs past the anchor's axes edge on the chosen
        # side — written by SubplotsAutoLayout._measure_one_group on each
        # settle iteration. We keep a local copy so the measurement pass
        # can subtract our own contribution from the cell's reservation
        # to derive the *pure* decoration base (title/xlabel/ylabel height
        # without the band) without a separate tightbbox measurement.
        self._band_contribution_mm: float = 0.0
        # anchor_ax pins position math/reactor to the anchor regardless
        # of self.ax swaps during add_* calls. For figure-anchored groups
        # it's the _GridAnchor proxy (decoration-aware). For axes-anchored
        # groups with a MULTI-axes scope (e.g. ``pp.legend(axes[0], side='top')``)
        # we pass the scope's _ScopeAnchor so along-edge geometry (edge
        # length, starting corner) comes from the scope's union — this is
        # the whole reason _ScopeAnchor exists. Single-axes scope still
        # uses the real Axes so vpad/interactions match pre-0.10 behaviour.
        # self.anchor remains the real Axes so auto_layout's
        # _find_ax_indices / per-cell reservation path keeps working.
        builder_ax = self.anchor if self._anchor_kind == "axes" \
                     else self._pick_builder_ax()
        if (
            self._anchor_kind == "axes"
            and self._scope_axes is not None
            and len(self._scope_axes) > 1
        ):
            builder_anchor_ax = self._scope_anchor
        else:
            builder_anchor_ax = self.anchor
        # In inside mode the LegendBuilder's side-dependent path (mm
        # cursor, outward gap, edge anchor) is bypassed by the
        # inside=True short-circuit, but the builder validates side=
        # against the 4 cardinal values in __init__ — so for the
        # only non-cardinal value we ever pass here (side='center'),
        # substitute a safe placeholder that the inside path ignores.
        builder_side = "right" if (self._inside and side == "center") else side
        self._builder = LegendBuilder(
            ax=builder_ax,
            anchor_ax=builder_anchor_ax,
            x_offset=x_offset,
            y_offset=y_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            max_width=max_width,
            external_to_axis=self._external_to_axis,
            side=builder_side,
            orientation=self._orientation,
        )
        # Register on the figure so plot functions can check claims.
        # Multiple groups can coexist on the same figure (each scoped to
        # a subset of axes via ``axes=``); they live in a list and the
        # "first to claim an axes/name pair" wins on conflict.
        fig = self.anchor.get_figure()
        groups = getattr(fig, "_publiplots_legend_groups", None)
        if groups is None:
            groups = []
            fig._publiplots_legend_groups = groups
        # Warn if another group already claims any axes in this scope
        # for the same entry name — the later group still gets
        # registered but won't evict/collect from the overlapping axes.
        for other in groups:
            overlap = self._scope_overlap(other)
            if overlap and self._collect_overlap(other):
                warnings.warn(
                    "pp.legend_group scope overlaps with an existing "
                    "group on this figure; the first group wins for "
                    "overlapping axes/entries. Pass disjoint ``axes=`` "
                    "or disjoint ``collect=`` to silence this warning.",
                    UserWarning,
                    stacklevel=3,
                )
                break
        groups.append(self)

        # Seamless "after" support: if the user attached the group AFTER
        # calling plot functions, each axes may already carry a
        # per-axis Legend artist that the group is about to render as a
        # shared one. Evict those to avoid duplicate (and stale) legend
        # rendering. Legends for entries NOT claimed by this group (e.g.,
        # inside legends for a different kind) stay untouched.
        self._evict_claimed_per_axis_legends()

        # Inside mode: blank the anchor cell at construction so the cell
        # reads as a clean tile regardless of whether _materialize ever
        # finds entries (collect=[] short-circuits _materialize before
        # the previous gating site, leaving the anchor's frame intact).
        # Idempotent: set_axis_off() is safe to call multiple times.
        if self._inside and self._clear_anchor:
            self.anchor.set_axis_off()

        # Alignment runs on each draw via the reactor hook so it uses
        # the current (possibly still-settling) figure geometry.
        # Absolute positioning → idempotent as geometry converges.
        # Connected here (not in _materialize) so it works whether the
        # user calls add_legend() directly or relies on auto-collect.
        if self._align != "start":
            self._connect_align_hook()

    def _iter_scope_axes(self):
        """Yield the axes this group collects from.

        ``None`` scope ⇒ every axes in ``fig._publiplots_axes``. Explicit
        ``axes=`` ⇒ just those.
        """
        if self._scope_axes is not None:
            for ax in self._scope_axes:
                yield ax
            return
        fig = self.anchor.get_figure()
        matrix = getattr(fig, "_publiplots_axes", None)
        if matrix is None:
            return
        for row in matrix:
            for ax in row:
                yield ax

    def _scope_contains(self, ax) -> bool:
        """True if this group collects from ``ax``."""
        if self._scope_axes is None:
            return True
        return any(ax is a for a in self._scope_axes)

    def _scope_overlap(self, other: "MultiAxesLegendGroup") -> bool:
        """True if ``self`` and ``other`` share any axes.

        Four cases:
        - Both ``_scope_axes=None`` → both auto-collect from the full
          grid, so they always compete.
        - Both explicit → compare by id().
        - One explicit, one ``_scope_axes=None`` (mixed):
          * If the ``None`` side is figure-anchored, it truly covers the
            whole grid → always overlap.
          * If the ``None`` side is axes-anchored (legacy ``pp.legend(anchor=ax)``),
            its physical region is just its anchor, so overlap only when
            the explicit scope includes that anchor. This lets
            ``pp.legend(axes[0])`` (scope=[axes[0]]) coexist with
            ``pp.legend(anchor=axes[1])`` (scope=None but anchored to a
            DIFFERENT axes) without a spurious overlap warning.
        """
        if self._scope_axes is None and other._scope_axes is None:
            return True
        if self._scope_axes is not None and other._scope_axes is not None:
            self_ids = {id(a) for a in self._scope_axes}
            return any(id(a) in self_ids for a in other._scope_axes)
        # Mixed: one has an explicit scope, the other is scope=None.
        explicit, implicit = (
            (self, other) if self._scope_axes is not None else (other, self)
        )
        if implicit._anchor_kind == "figure":
            return True
        # implicit is axes-anchored with scope=None → physical target
        # is just the anchor cell.
        return any(a is implicit.anchor for a in explicit._scope_axes)

    def _collect_overlap(self, other: "MultiAxesLegendGroup") -> bool:
        """True if ``self`` and ``other`` could claim the same entry name."""
        if self._collect is None or other._collect is None:
            return True
        return bool(set(self._collect) & set(other._collect))

    def _evict_claimed_per_axis_legends(self) -> None:
        """Remove per-axis Legend artists that the group will render itself.

        Walks every axes in this group's scope (``self._scope_axes`` or
        the full grid when unset). For each stashed LegendEntry this
        group claims, matches the axes' Legend children by title text
        and removes them — plus unregisters their ``LayoutReactor``
        registrations so the reactor stops repositioning ghost artists
        and ``SubplotsAutoLayout`` shrinks the per-column/row
        reservation on the next settle pass.
        """
        from matplotlib.legend import Legend

        reactor = self._builder._reactor
        claimed_names = set()
        scope_axes = list(self._iter_scope_axes())
        for ax in scope_axes:
            for entry in get_entries(ax):
                if self.claims(entry.name):
                    claimed_names.add(entry.name)

        if not claimed_names:
            return

        to_unregister = []
        for ax in scope_axes:
            for child in list(ax.get_children()):
                if not isinstance(child, Legend):
                    continue
                title = child.get_title().get_text()
                if title not in claimed_names:
                    continue
                child.remove()
                if ax.legend_ is child:
                    ax.legend_ = None
                to_unregister.append(child)

        if to_unregister:
            ids = {id(a) for a in to_unregister}
            reactor._registrations = [
                r for r in reactor._registrations if id(r.artist) not in ids
            ]
            # Also purge from the builders' elements lists (per-axis
            # builders stored on ax._legend_builder) so duplicate
            # re-adds don't resurrect them.
            for ax in scope_axes:
                builder = getattr(ax, "_legend_builder", None)
                if builder is None:
                    continue
                builder.elements = [
                    (k, a) for (k, a) in builder.elements if id(a) not in ids
                ]

    def _connect_align_hook(self) -> None:
        """Wrap LayoutReactor._refresh_all so our alignment callback
        runs after every reactor refresh. One wrap per reactor even
        when multiple groups share it; each group appends its own
        callback.

        The wrap skips the callback phase while already running one
        (``reactor._aligning_in_progress``). Each callback's own
        ``_apply_along_alignment`` call triggers a nested
        ``_refresh_all`` to propagate the new mm offsets to matplotlib;
        without the shared flag that nested call would re-enter every
        *other* group's callback, so two figure-anchored groups would
        cascade and mutate each other's registrations into a broken
        state. The per-group ``self._aligning`` flag alone can't catch
        this because each group is a different ``self``.
        """
        if self._align_connected:
            return
        reactor = self._builder._reactor
        if not hasattr(reactor, "_align_callbacks"):
            reactor._align_callbacks = []
            reactor._aligning_in_progress = False
            _orig_refresh = reactor._refresh_all

            def _refresh_then_align():
                result = _orig_refresh()
                if reactor._aligning_in_progress:
                    return result
                reactor._aligning_in_progress = True
                try:
                    for cb in reactor._align_callbacks:
                        cb()
                finally:
                    reactor._aligning_in_progress = False
                return result

            reactor._refresh_all = _refresh_then_align
        reactor._align_callbacks.append(self._on_draw_align_cb)
        self._align_connected = True

    def _pick_builder_ax(self) -> Axes:
        """Pick a real axes from the figure to attach Legend artists to.

        Figure-anchored groups use ``_GridAnchor`` only for position
        math; the actual matplotlib ``Legend`` artist still needs a
        real Axes parent. Default to the corner cell closest to the
        legend's edge so hit-testing lines up with where the user sees
        the band sitting.
        """
        fig = self.anchor.get_figure()
        matrix = getattr(fig, "_publiplots_axes", None)
        if matrix is None:
            return fig.axes[0]
        if self._side == "right":
            return matrix[0][-1]
        if self._side == "left":
            return matrix[0][0]
        if self._side == "bottom":
            return matrix[-1][0]
        return matrix[0][0]  # "top"

    def claims(self, name: str) -> bool:
        """True if the group will render an entry with this name."""
        if self._collect is None:
            return True
        return name in self._collect

    def _materialize(self) -> None:
        """Collect stashed entries from every grid axes and render them.

        Called by SubplotsAutoLayout during the settle pass. The first
        call that finds at least one matching entry renders them and
        marks the group materialized; earlier calls that see nothing
        (the group was constructed BEFORE its scope's plots ran) are
        no-ops so the group can materialize on a later draw once
        entries exist. Once materialized, subsequent calls short-circuit
        — rendering is a one-shot operation.
        """
        if self._materialized:
            return

        # collect=[] is the explicit "skip auto-collection" shortcut used
        # by callers who want to add manual handles via add_legend() only
        # (replaces the old pp.legend(ax, auto=False) idiom). Short-circuit
        # here so repeated draws don't keep re-scanning the scope.
        if self._collect is not None and len(self._collect) == 0:
            self._materialized = True
            return

        # Gather all entries per (name, kind) across the group's scope.
        by_key = {}   # (name, kind) -> list[LegendEntry]
        order = []    # (name, kind) in collection order (first-seen)
        for ax in self._iter_scope_axes():
            for entry in get_entries(ax):
                if self._collect is not None and entry.name not in self._collect:
                    continue
                key = (entry.name, entry.kind)
                if key not in by_key:
                    by_key[key] = []
                    order.append(key)
                by_key[key].append(entry)

        if not by_key:
            # Nothing to render yet — the plot functions haven't stashed
            # anything in this group's scope. Leave _materialized=False so
            # a future draw (after the stashing) can try again.
            return
        self._materialized = True

        if self._collect is not None:
            # Stable sort by the user's collect order; ties (same name with
            # different kinds) stay in the order they were encountered.
            order.sort(key=lambda k: (self._collect.index(k[0]), 0))

        for key in order:
            merged = self._merge_entries(key, by_key[key])
            self._render_entry(merged)

    def _merge_entries(self, key, entries):
        """Merge a list of same-(name, kind) entries into one.

        For categorical entries, the union of labels across axes is
        preserved in first-seen order; handles come from whichever
        axes stashed each label first. A mismatched signature on
        shared labels triggers a single warning (first-occurrence
        wins on conflict, but the merge itself is additive — no
        labels are dropped).

        Continuous-hue entries (ScalarMappable) cannot be meaningfully
        merged; first-occurrence wins and we warn if subsequent
        entries differ.
        """
        if len(entries) == 1:
            return entries[0]

        first = entries[0]
        # Continuous hue: merging a ScalarMappable is ill-defined
        # (different cmaps / norms don't concatenate). Keep the first
        # and warn if anyone else disagrees.
        if is_continuous_hue(first.handles):
            for other in entries[1:]:
                if other.signature != first.signature and not self._warned_mismatch:
                    warnings.warn(
                        f"continuous legend entry {first.name!r} differs "
                        "between axes; group uses first occurrence",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_mismatch = True
                    break
            return first

        # Categorical: union of (label, handle) pairs, first-seen wins
        # for each label. Track whether any axes had a different handle
        # for a label we already have — that's the "signature mismatch"
        # case where we warn but keep the first handle.
        merged_labels = []
        merged_handles = []
        label_to_handle_repr = {}
        additive_merge = False
        conflict = False
        for entry in entries:
            for lab, hand in zip(entry.labels, entry.handles):
                if lab not in label_to_handle_repr:
                    merged_labels.append(lab)
                    merged_handles.append(hand)
                    label_to_handle_repr[lab] = _handle_repr(hand)
                elif label_to_handle_repr[lab] != _handle_repr(hand):
                    conflict = True
            if len(merged_labels) > len(entry.labels):
                additive_merge = True

        if additive_merge and not self._warned_mismatch:
            warnings.warn(
                f"legend entry {first.name!r} ({first.kind}) merged across "
                f"axes with different label sets; union rendered.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_mismatch = True
        if conflict and not self._warned_mismatch:
            warnings.warn(
                f"legend entry {first.name!r} ({first.kind}) has inconsistent "
                "handles for the same label across axes; first occurrence used.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_mismatch = True

        return LegendEntry.build(
            name=first.name,
            kind=first.kind,
            handles=merged_handles,
            labels=merged_labels,
        )

    def _on_draw_align_cb(self) -> None:
        """Post-reactor-refresh alignment hook. Runs inside each draw
        after LayoutReactor has repositioned every registration; our
        alignment override fires on the registrations this group owns,
        then those overrides reach the canvas via the reactor's next
        refresh. Guarded against re-entrancy (measuring triggers draws).
        """
        if self._aligning:
            return
        self._aligning = True
        try:
            self._apply_along_alignment()
            # Re-run reactor refresh with the updated mm_y_from_top
            # values so matplotlib renders the new positions in this
            # draw (not the next one).
            self._builder._reactor._refresh_all()
        finally:
            self._aligning = False

    def _apply_along_alignment(self) -> None:
        """Re-position all placed reactor registrations along the anchor edge
        to honor ``self._align``.

        Each row (legends sharing an outward offset) is aligned
        independently so a wrapped block stays visually balanced. We
        recompute each registration's **absolute** along-edge offset
        (``mm_y_from_top`` — historical field name, semantically
        "along-edge mm from the starting corner") from the known legend
        extents, then re-run the reactor so the legends snap to the new
        figure-fraction positions.

        The starting corner of the along-edge axis is ``ax_pos.y1`` for
        side=right/left (top of anchor) and ``ax_pos.x0`` for
        side=top/bottom (left of anchor). "Along-edge mm from start"
        increases as we move AWAY from that corner.
        """
        if self._align == "start":
            return
        if not self._builder.elements:
            return

        reactor = self._builder._reactor
        element_ids = {id(a) for _, a in self._builder.elements}
        regs = [r for r in reactor._registrations if id(r.artist) in element_ids]
        if not regs:
            return

        orient = self._orientation
        edge_length_mm = self._builder._get_edge_length()
        gap_mm = self._builder._layout.gap

        # Group regs into rows sharing the same outward offset.
        rows = {}
        for reg in regs:
            key = round(reg.mm_x_from_right, 3)
            rows.setdefault(key, []).append(reg)

        for row_regs in rows.values():
            # Measure each legend's along-edge extent.
            extents = []
            for reg in row_regs:
                w, h = self._builder._measure_object_dimensions(reg.artist)
                extents.append(w if orient == "horizontal" else h)
            total = sum(extents) + gap_mm * (len(row_regs) - 1)
            if total >= edge_length_mm:
                # Block already fills the edge — no room to align.
                continue
            if self._align == "center":
                start = (edge_length_mm - total) / 2
            elif self._align == "end":
                start = edge_length_mm - total - self._builder._layout.vpad
            else:
                start = self._builder._layout.vpad
            cursor = start
            for reg, extent in zip(row_regs, extents):
                reg.mm_y_from_top = cursor
                cursor += extent + gap_mm

        reactor._refresh_all()

    def _set_decoration_offset(self, mm: float) -> None:
        """Bake ``mm`` into every reactor registration this group owns.

        Called by ``SubplotsAutoLayout._measure_one_group`` inside the
        layout measurement pass — a known-safe context (the reactor is
        mid-refresh but this is not itself a reactor callback). The next
        ``_update_artist_anchor`` call picks up the new value via
        ``reg.mm_outward_decoration_offset`` and repositions the artist past the
        anchor's decorations on the chosen side.
        """
        if not self._builder.elements:
            return
        reactor = self._builder._reactor
        element_ids = {id(a) for _, a in self._builder.elements}
        for reg in reactor._registrations:
            if id(reg.artist) in element_ids:
                reg.mm_outward_decoration_offset = mm

    def _render_entry(self, entry: LegendEntry) -> None:
        """Route to add_legend (categorical) or add_colorbar (continuous)."""
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            mappable = entry.handles[0]
            self.add_colorbar(mappable=mappable, label=entry.name)
        else:
            self.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )

    def _default_target_ax(self) -> Axes:
        """Axes to attach a Legend artist to when no explicit ax= is passed.

        For axes-anchored groups, that's the anchor. For figure-anchored
        groups the anchor is a ``_GridAnchor`` proxy (not a real Axes);
        the builder's pre-picked corner cell is used instead.
        """
        if self._anchor_kind == "axes":
            return self.anchor
        return self._builder.ax  # corner cell picked in __init__

    def add_legend(
        self,
        handles: List,
        label: str = "",
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Legend:
        """Add a legend to the shared band.

        The artist is attached to ``ax`` (defaults to a sensible corner
        cell for figure-anchored groups, or the anchor for axes-anchored);
        position is always computed against the anchor's chosen edge.

        For inside-mode groups (``inside=True`` on the factory), this
        method auto-injects ``inside=True`` + ``loc=`` derived from the
        group's ``side`` / ``align`` so that manual handle-add calls
        (e.g. ``band.add_legend(..., ncol=2)`` after ``collect=[]``)
        render through the same axes-relative path as the auto-collect
        branch. Explicit user kwargs win.
        """
        target_ax = ax if ax is not None else self._default_target_ax()
        if self._inside:
            kwargs.setdefault("inside", True)
            kwargs.setdefault(
                "loc", _inside_loc_from_side_align(self._side, self._align),
            )
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            legend = self._builder.add_legend(handles=handles, label=label, **kwargs)
        finally:
            self._builder.ax = original_ax
        return legend

    def add_colorbar(
        self,
        mappable: Optional[ScalarMappable] = None,
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Colorbar:
        """Add a colorbar to the shared band. See add_legend for ax semantics."""
        target_ax = ax if ax is not None else self._default_target_ax()
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            cbar = self._builder.add_colorbar(mappable=mappable, **kwargs)
        finally:
            self._builder.ax = original_ax
        return cbar


_SIDE_SENTINEL = object()


def legend(
    axes=None,
    collect: Optional[Sequence[str]] = None,
    *,
    side=_SIDE_SENTINEL,
    anchor: Optional[Axes] = None,
    figure: Optional[Figure] = None,
    orientation: str = "auto",
    align: str = "auto",
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: Optional[float] = None,
    max_width: Optional[float] = None,
    # PR 4: grid-scope kwargs (additive; all default None → existing behavior)
    rows: Optional[_RowColSpec] = None,
    cols: Optional[_RowColSpec] = None,
    span: Optional[str] = None,
    ax: Optional[Sequence[Axes]] = None,
    inside: bool = False,
    clear_anchor=_SIDE_SENTINEL,
) -> MultiAxesLegendGroup:
    """Create a publication-ready legend for one axes, a subset, or the full figure.

    Unified replacement for the legacy ``pp.legend_group`` API (pre-0.10).
    The first positional is the **scope** — which axes to collect legend
    entries from — and the anchor is derived automatically.

    Parameters
    ----------
    axes : Axes, sequence of Axes, or None
        Scope of the legend.

        - A single ``Axes``: per-axes legend pinned to that axes
          (lives inside the axes tightbbox).
        - A list/array of ``Axes``: shared band over the scope's
          bounding rect (lives outside the axes, measured as an
          overhang by ``SubplotsAutoLayout``).
        - ``None`` (default): full-figure band over the whole
          subplot grid.
    collect : sequence of str, optional
        Names of entries to collect from the scope's stashed
        ``LegendEntry`` objects. ``None`` collects everything; an
        empty list ``[]`` skips auto-collection entirely (equivalent
        to the old ``pp.legend(ax, auto=False)`` idiom — use it when
        adding manual handles via ``add_legend(...)``).
    side : {"right", "bottom", "left", "top"}, default "right"
        Which edge of the scope the band grows outward from.
    anchor : Axes, optional
        Advanced override: pin the band's geometry to a specific
        axes' edge, while still collecting entries from the ``axes``
        scope. Rarely needed — defaults to the scope's bounding rect.
    figure : Figure, optional
        Figure to attach a full-figure band to. Defaults to
        ``plt.gcf()``.
    orientation, align, x_offset, y_offset, gap, column_spacing, vpad, max_width
        Same meaning as on :class:`MultiAxesLegendGroup`.

    Examples
    --------
    >>> pp.legend(ax)                          # per-axes legend
    >>> pp.legend(axes[0])                     # row 0 band
    >>> pp.legend(axes[:, 0], side='left')     # column-0 band
    >>> pp.legend(side='right')                # full-figure right band
    >>> pp.legend(side='bottom', collect=['group'])

    Returns
    -------
    MultiAxesLegendGroup
        The constructed group. Can be further customized via
        ``group.add_legend(...)`` / ``group.add_colorbar(...)``.

    Notes
    -----
    **Positional vs. keyword asymmetry on a single axes.**  There is a
    deliberate semantic split between the two spellings of "legend for
    one axes":

    - ``pp.legend(ax)`` (positional) — **internal** per-axes legend;
      the legend counts against ``ax.get_tightbbox()`` and the axes
      reserves its own space (like pre-0.10 ``pp.legend(ax)``).
    - ``pp.legend(anchor=ax)`` (keyword) — **external** band pinned to
      that axes' edge; the legend is measured as an overhang by
      :class:`SubplotsAutoLayout` and grows the figure's per-cell
      reservation (like pre-0.10 ``pp.legend_group(anchor=ax)``).

    Both render the same visually for a single isolated axes. They
    differ under ``pp.subplots`` where the distinction controls which
    reservation the layout reactor grows.
    """
    # ------------------------------------------------------------------
    # inside= validation + side default
    # ------------------------------------------------------------------
    # ``inside=True`` flips the render mode: the legend lives inside the
    # anchor's rectangle (matplotlib loc=) instead of overhanging the
    # anchor's edge as a band. Default ``side='left', align='start'``
    # makes the legend tile a visual continuation of the band-mode
    # ``side='right', align='start'`` recipe (legend hugs the inner-left
    # edge of the legend tile, against the divide between plots and
    # legend).
    if inside:
        if anchor is None:
            raise ValueError(
                "pp.legend: inside=True requires anchor=<Axes> (the cell "
                "to render the legend into). Got anchor=None."
            )
        if not isinstance(anchor, Axes):
            raise ValueError(
                "pp.legend: inside=True requires anchor to be a single "
                f"Axes; got {type(anchor).__name__}."
            )
        # Inside-mode default for side. Detect "user did not pass side="
        # via the sentinel so an explicit ``side='right'`` still wins.
        side = "left" if side is _SIDE_SENTINEL else side
    else:
        if clear_anchor is not _SIDE_SENTINEL:
            raise ValueError(
                "pp.legend: clear_anchor= is only meaningful with "
                "inside=True."
            )
        # Out of inside mode: side= keeps its existing default of 'right'.
        side = "right" if side is _SIDE_SENTINEL else side
    # Resolve clear_anchor= to its canonical bool. Default True in
    # inside mode; ignored otherwise (already validated above).
    clear_anchor_resolved = (
        True if clear_anchor is _SIDE_SENTINEL else bool(clear_anchor)
    )

    # PR 4: if any of the new grid-scope kwargs are set, resolve them up-front
    # to a list-of-axes scope (or None for figure-level), then fall through to
    # the existing resolution logic by setting `axes`.
    if rows is not None or cols is not None or span is not None or ax is not None:
        # span='row'/'col' is the SOLE exception to "new kwargs are mutually
        # exclusive with the positional `axes=` arg" — those two span values
        # REQUIRE a positional anchor, so they consume `axes=ax` instead of
        # raising on it.
        is_positional_anchor_span = (
            span in ("row", "col")
            and isinstance(axes, Axes)
            and rows is None and cols is None and ax is None
        )

        if axes is not None and not is_positional_anchor_span:
            raise ValueError(
                "pp.legend: legacy positional `axes=` is mutually exclusive "
                "with the new `rows=`/`cols=`/`span=`/`ax=` kwargs. Use one "
                "addressing mode at a time."
            )

        resolver_fig = figure if figure is not None else plt.gcf()

        if is_positional_anchor_span:
            # Expand against the anchor; consume `axes` and continue down the
            # list-of-axes branch.
            expanded = _expand_span_with_anchor(resolver_fig, axes, span)
            axes = expanded
        else:
            if span in ("row", "col") and not isinstance(axes, Axes):
                raise ValueError(
                    f"pp.legend: span={span!r} requires a positional Axes "
                    "anchor (e.g. `pp.legend(axes[0,1], span='row')`)."
                )
            resolved_scope = _resolve_grid_scope(
                resolver_fig, rows=rows, cols=cols, span=span, ax=ax,
            )
            if resolved_scope is not None:
                # Drift guard: if `figure=` was explicit AND the resolved axes
                # belong to a different figure, bail rather than letting the
                # downstream group silently switch figures off the resolved
                # axes' get_figure(). The `ax=` path is the most common way
                # to hit this — caller passes ax= from one figure but figure=
                # naming another.
                if figure is not None:
                    for a in resolved_scope:
                        if a.get_figure() is not figure:
                            raise ValueError(
                                "pp.legend: resolved axes belong to a different "
                                "Figure than the one passed via `figure=`. "
                                "Either drop `figure=` or pass axes from that "
                                "figure."
                            )
                axes = resolved_scope
            # else: span='fig' or all-None → leave axes=None.

    # Resolve anchor + axes per the rules:
    #   axes=None, anchor=None  -> figure-level (whole grid)
    #   axes=None, anchor=ax    -> scope=[ax]   (back-compat: old legend_group(anchor=ax),
    #                                            preserves external-band semantics — no flip)
    #   axes=ax, anchor=None    -> scope=[ax]   (single-axes → sets external_to_axis=False)
    #   axes=list, anchor=None  -> list         (anchor defaults to list[0])
    #   axes=list, anchor=ax    -> list         (anchor as explicit override)
    resolved_anchor = anchor
    resolved_axes = axes
    # Track whether the caller triggered single-axes scope via the new-API
    # `axes=ax` form (which opts into the external_to_axis=False flip) vs.
    # the legacy `anchor=ax` form (which preserves the pre-0.10 external
    # band behaviour).
    axes_triggered_single_scope = False

    if axes is None and anchor is None:
        # figure-level → anchor stays None, axes stays None
        pass
    elif axes is None and anchor is not None:
        # Bare anchor=ax → preserve legacy pp.legend_group(anchor=ax)
        # external-band semantics. Scope stays figure-wide; the band pins
        # to `anchor`'s edge and is measured as an overhang.
        pass
    elif isinstance(axes, Axes) and anchor is None:
        # Single axes: anchor to that axes; set resolved_axes=[axes] so
        # the group has an EXPLICIT single-axes scope (not ``None``,
        # which means "full grid"). The explicit scope is what lets
        # _scope_overlap correctly see two per-axes legends on different
        # axes as non-overlapping; with scope=None the overlap heuristic
        # false-positives and spurious warnings fire when e.g.
        # pp.legend(axes[0]) and pp.legend(anchor=axes[1]) coexist.
        resolved_anchor = axes
        resolved_axes = [axes]
        axes_triggered_single_scope = True
    elif isinstance(axes, Axes) and anchor is not None:
        # axes=ax, anchor=other_ax → explicit anchor override, single-axes scope.
        resolved_axes = [axes]
        # resolved_anchor already equals anchor
        axes_triggered_single_scope = True
    else:
        # axes is a list/array/tuple. anchor may be explicit or None.
        resolved_axes = list(axes)
        if resolved_anchor is None and resolved_axes:
            # Default the anchor to the first scoped axes so the band has
            # a concrete edge to pin to. Matches the implicit choice the
            # current MultiAxesLegendGroup makes when anchor=<Axes> is
            # passed alongside a full axes= list.
            resolved_anchor = resolved_axes[0]

    group = MultiAxesLegendGroup(
        anchor=resolved_anchor,
        collect=collect,
        side=side,
        figure=figure,
        orientation=orientation,
        align=align,
        axes=resolved_axes,
        x_offset=x_offset,
        y_offset=y_offset,
        gap=gap,
        column_spacing=column_spacing,
        vpad=vpad,
        max_width=max_width,
        inside=inside,
        clear_anchor=clear_anchor_resolved,
    )

    # Flip external_to_axis for single-axes scope triggered via the new
    # `axes=ax` form. The class default is True (preserving pre-0.10
    # pp.legend_group(anchor=ax) "external band" semantics — which the
    # legacy `anchor=ax` kwarg form still opts into); pp.legend(ax) /
    # pp.legend(axes=ax) want the opposite — an in-frame legend that
    # lives inside ax.tightbbox and is counted via _side_extent, not as
    # an overhang. Must mutate BOTH the group attribute (read by
    # SubplotsAutoLayout._measure_one_group's commit-3 guard) AND the
    # LegendBuilder's forwarded copy (read by every reactor registration
    # spawned from add_legend/add_colorbar). The factory returns the
    # group BEFORE any add_* / _materialize call, so every subsequent
    # registration picks up the False flag.
    if axes_triggered_single_scope:
        group._external_to_axis = False
        group._builder._external_to_axis = False

    return group


def _get_or_create_per_axes_group(ax: Axes) -> MultiAxesLegendGroup:
    """Return (or create) the single-axes legend group cached on ``ax``.

    Used by ``render_entries`` to funnel every plot function's legend
    output through a per-axes ``MultiAxesLegendGroup``, so successive
    plot calls stack their legend entries into the same layout cursor.
    Flipped to ``external_to_axis=False`` so the legend is measured by
    ``ax.get_tightbbox()`` (matches the pre-0.10 ``pp.legend(ax)``
    behaviour).

    ``collect=[]`` is passed so the group does NOT auto-collect stashed
    entries during its ``_materialize`` pass — ``render_entries`` is the
    sole producer of ``add_legend``/``add_colorbar`` calls on the
    underlying builder. Without this, the group would scan the axes'
    stashed entries during draw and re-add them on top of the ones
    ``render_entries`` already added, producing duplicate legends.

    Also preserves the ``ax._legend_builder`` alias so the existing
    ``_evict_claimed_per_axis_legends`` read path keeps working.
    """
    existing = getattr(ax, "_legend_group", None)
    if existing is not None:
        return existing
    # collect=[] — see docstring; render_entries owns the add_legend calls.
    group = legend(ax, collect=[])
    ax._legend_group = group
    ax._legend_builder = group._builder  # back-compat alias
    return group