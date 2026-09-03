"""
Draw-event hook that keeps declared axes sizes fixed while the figure
grows to fit auto-measured decorations.

Reservations are per-row (title_space, xlabel_space) or per-column
(ylabel_space, right). Measurement excludes LayoutReactor-managed
artists flagged external_to_axis (e.g., pp.legend_group) — those are
handled by the reactor's own anchoring geometry plus the user's
legend_column reservation, not by axis-level tightbbox.

Cooperates with LayoutReactor (utils/layout_reactor.py): both react to
draw_event, but SubplotsAutoLayout is registered first (during
pp.subplots()) and therefore fires first, so LayoutReactor sees the
repositioned axes and re-anchors legends correctly.
"""

import warnings
from typing import Dict, FrozenSet, Optional, Set, Tuple

import matplotlib as mpl

from publiplots.layout.figure_layout import FigureLayout


_MM2INCH = 1 / 25.4
_UPDATE_THRESHOLD_MM = 0.1
_ALL_SIDES = {
    "title_space", "xlabel_space", "ylabel_space", "right",
    "legend_column", "legend_band_bottom", "legend_band_top", "legend_band_left",
    "suptitle_space",
}
# Cap on settle() draws; 1-3 is typical.
_MAX_CONVERGENCE_ITERS = 5

# Floor on the residual settle() reports as a non-convergence, in the same
# millimetres _needs_update compares. Well above the tolerance, because a
# residual just over the tolerance is not necessarily a bug and a bug is not
# a residual just over the tolerance:
#
#   * settle() runs inside the print_figure wrapper BEFORE matplotlib swaps
#     fig.dpi to savefig's, and the reactor is frozen for the render, so the
#     check always evaluates at figure.dpi. Across 25 layouts saved three
#     times each at 72/100/150/300/600 dpi, the residual the next settle()
#     saw was 0.00 mm every time (375/375) — a render at another dpi leaves
#     no hysteresis behind. Were the check ever to run at the render's own
#     dpi, though, text metrics are not dpi-invariant and the same corpus
#     measured up to 0.76 mm of purely dpi-induced drift (0.1 mm is roughly
#     one 600-dpi pixel, so the tolerance alone has no margin at all). This
#     floor clears that ceiling.
#   * The layouts that actually ship broken clear it by two orders of
#     magnitude: 67.24 mm/draw for the #230 inner-anchor band, 44-74 mm/draw
#     for the #244 inset_axes scope.
_NONCONVERGENCE_WARN_MM = 1.0


class LayoutConvergenceWarning(UserWarning):
    """A figure's layout never settled, so its saved size is not reproducible.

    Emitted at most once per figure by :meth:`SubplotsAutoLayout.settle`.
    Subclasses ``UserWarning`` deliberately: every filter a user already has
    for publiplots' other warnings keeps catching it, while a known-benign
    case can be silenced on its own with

    >>> warnings.filterwarnings("ignore", category=pp.LayoutConvergenceWarning)

    without also muting the plot-level warnings that share ``UserWarning``.
    """


# side_name -> (axis_kind, bbox_fn)
#   axis_kind: "row" (result length == nrows) or "col" (length == ncols)
#   bbox_fn:   float (ax_bbox, tight_bbox) -> px
_SIDE_CALCULATORS = {
    "title_space":  ("row", lambda ax_bb, t: t.y1 - ax_bb.y1),
    "xlabel_space": ("row", lambda ax_bb, t: ax_bb.y0 - t.y0),
    "ylabel_space": ("col", lambda ax_bb, t: ax_bb.x0 - t.x0),
    "right":        ("col", lambda ax_bb, t: t.x1 - ax_bb.x1),
}


class SubplotsAutoLayout:
    """Per-figure draw-event listener that resizes the figure to fit decorations."""

    def __init__(
        self,
        fig,
        layout: FigureLayout,
        locked: Set[str],
        locked_positions: Optional[Dict[str, FrozenSet[int]]] = None,
    ):
        """
        Parameters
        ----------
        locked : set of str
            Sides whose reservation is FULLY locked (every position pinned).
            Names are auto-measurable side names (``title_space``,
            ``xlabel_space``, ``ylabel_space``, ``right``) and the legend /
            suptitle scalars enumerated in ``_ALL_SIDES``.
        locked_positions : dict[str, frozenset[int]], optional
            Per-position locks for the four auto-measurable per-cell sides.
            ``locked_positions["xlabel_space"] = frozenset({0})`` means
            ``xlabel_space[0]`` is pinned to its initial value while the
            remaining positions auto-measure. A side appearing in ``locked``
            takes precedence (whole-side lock). Empty / missing entries
            mean "no per-position lock".
        """
        self._fig = fig
        self._layout = layout
        self._locked = set(locked)
        self._locked_positions: Dict[str, FrozenSet[int]] = {
            side: frozenset(idxs) for side, idxs in (locked_positions or {}).items() if idxs
        }
        self._updating = False
        # One SubplotsAutoLayout per figure, so this flag makes the
        # non-convergence warning once-per-figure rather than once-per-save:
        # saving the same broken figure in a loop reports it once.
        self._nonconvergence_warned = False

        fig._publiplots_layout = layout
        fig._publiplots_auto_layout = self

        if _ALL_SIDES.issubset(self._locked):
            self._cid = None
        else:
            self._cid = fig.canvas.mpl_connect("draw_event", self._on_draw)

        # Wrap the render-to-file entry point so that by the time it
        # renders, the figure has been resized to fit its current
        # decorations -- and so that no resize can happen *during* that
        # render. See _install_render_wrapper.
        self._install_render_wrapper()

    def _install_render_wrapper(self) -> None:
        """Settle before a render-to-file, and freeze the reactor during it.

        draw_event fires AFTER the renderer has written its buffer, so a
        resize from ``_on_draw`` during the output render is worse than
        merely late. Agg keys its renderer cache on
        ``figure.bbox.size``; a mid-draw ``set_size_inches`` invalidates
        the renderer that was just drawn into, so the bytes handed to the
        file writer come from a freshly allocated, never-drawn (i.e.
        fully transparent) buffer. The result is a blank image.

        ``settle()`` is meant to make that resize unnecessary, but it
        cannot make it impossible: measurements are taken from text
        metrics, which are not dpi-invariant, so a layout that has
        converged at ``figure.dpi`` can still measure >
        ``_UPDATE_THRESHOLD_MM`` differently at ``savefig``'s dpi and
        trigger exactly that mid-render resize. Freezing the reactor for
        the duration of the render closes the hole for good: whatever
        happens, the buffer the writer reads is the buffer that was
        drawn. The frozen render is preceded by ``settle()``, so the
        layout it captures is the converged one.

        The hook is ``canvas.print_figure`` rather than ``fig.savefig``
        because every render-to-file path funnels through it
        (``fig.savefig``, ``plt.savefig``, IPython's inline display).
        """
        fig = self._fig
        if getattr(fig, "_publiplots_render_wrapped", False):
            return
        canvas = fig.canvas
        original_print_figure = canvas.print_figure

        def _wrapped_print_figure(*args, **kwargs):
            self.settle()
            was_updating = self._updating
            self._updating = True
            try:
                return original_print_figure(*args, **kwargs)
            finally:
                self._updating = was_updating

        canvas.print_figure = _wrapped_print_figure
        fig._publiplots_render_wrapped = True

    def settle(self) -> None:
        """Drive the layout to convergence without leaving stale state.

        Runs canvas.draw() up to a small number of times. Each draw
        fires our _on_draw, which measures and (if needed) resizes.
        Once _needs_update returns False, we stop. This is the safe
        settlement primitive — unlike in-event iteration, each draw is
        a complete matplotlib pass with its own renderer, avoiding the
        reentrancy hazards of draw_without_rendering inside draw_event.

        Exhausting the cap used to return silently, which is how #230
        (a band anchored to an inner axes, +67.24 mm on every draw) and
        the twinx runaway (+115.56 mm) reached published releases: the
        layout knew it had given up and told nobody, and the only
        symptom the user saw was ``savefig`` producing a different size
        each time it was called. On exhaustion we now warn once per
        figure — see :meth:`_warn_not_converged`.

        Scope: the warning reaches a user wherever ``settle()`` runs,
        which is the ``print_figure`` wrapper — so on ``pp.savefig`` and
        on inline display, the two paths where a non-convergent layout
        actually produces a wrong file. Driving ``fig.canvas.draw()`` in
        a loop by hand never calls ``settle()`` and so never warns
        (measured: six draws of the #244 runaway, zero warnings). That
        is deliberate — a single draw is not a convergence claim — but
        it does mean the check is on the output path, not on every
        draw.
        """
        fig = self._fig
        sizes = []
        measured: Dict[str, Tuple[float, ...]] = {}
        for _ in range(_MAX_CONVERGENCE_ITERS):
            fig.canvas.draw()
            sizes.append(tuple(fig.get_size_inches()))
            measured = self._measure()
            if not self._needs_update(measured):
                return
        self._warn_not_converged(measured, sizes)

    def _worst_residual(
        self, measured: Dict[str, Tuple[float, ...]]
    ) -> Tuple[Optional[str], float]:
        """The largest deviation ``_needs_update`` sees, as ``(field, mm)``.

        Expressed in exactly the terms ``_needs_update`` compares: the
        absolute millimetre difference between a freshly measured
        reservation and the one ``self._layout`` currently holds, taken
        per position for the per-row / per-column tuple sides so the
        label can name the offending cell (``right[1]``) and not just the
        side. A length mismatch — the grid itself changed shape — has no
        finite residual and reports as ``inf``.
        """
        field: Optional[str] = None
        residual = 0.0
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if side in self._SCALAR_SIDES:
                pairs = ((None, new_val, current),)
            else:
                if len(new_val) != len(current):
                    return (
                        f"{side} (length {len(current)} -> {len(new_val)})",
                        float("inf"),
                    )
                pairs = tuple(
                    (i, nv, cv) for i, (nv, cv) in enumerate(zip(new_val, current))
                )
            for idx, nv, cv in pairs:
                delta = abs(nv - cv)
                if delta > residual:
                    residual = delta
                    field = side if idx is None else f"{side}[{idx}]"
        return field, residual

    def _warn_not_converged(self, measured, sizes) -> None:
        """Report an exhausted convergence budget, once per figure.

        Gated on the residual's *magnitude*, not on its growth. That is a
        measurement, not a preference: the two layouts known to diverge
        hold a residual that is flat (#230 reports 67.2372 mm on
        ``right[0]`` on every single draw, to the fourth decimal) or
        non-monotonic (#244's inset_axes scope runs 44.3 → 68.9 → 74.3 →
        68.5 → 61.8 → 57.5 mm) while the *figure* grows without bound
        underneath them. A "residual is growing" test would catch neither.
        What separates them from noise is scale — see
        ``_NONCONVERGENCE_WARN_MM``.

        The message names the field and the residual because "layout did
        not converge" is not something a user can act on, and quotes the
        figure's growth over the capped draws because that is the symptom
        they actually hit: a figure whose saved size is not reproducible.
        """
        if self._nonconvergence_warned:
            return
        field, residual = self._worst_residual(measured)
        if field is None or residual < _NONCONVERGENCE_WARN_MM:
            return
        self._nonconvergence_warned = True

        (w0, h0), (w1, h1) = sizes[0], sizes[-1]
        dw, dh = (w1 - w0) * 25.4, (h1 - h0) * 25.4
        if abs(dw) >= _UPDATE_THRESHOLD_MM or abs(dh) >= _UPDATE_THRESHOLD_MM:
            growth = (
                f"over those draws the figure went from {w0 * 25.4:.2f} x "
                f"{h0 * 25.4:.2f} mm to {w1 * 25.4:.2f} x {h1 * 25.4:.2f} mm "
                f"({dw:+.2f} x {dh:+.2f} mm) and had not stopped"
            )
        else:
            growth = (
                f"the figure size held at {w1 * 25.4:.2f} x {h1 * 25.4:.2f} mm, "
                f"so the reservation is oscillating rather than running away"
            )
        warnings.warn(
            f"publiplots: this figure's layout did not converge after "
            f"{_MAX_CONVERGENCE_ITERS} draws. The reservation furthest from "
            f"settling is {field}, still moving {residual:.2f} mm per draw "
            f"(tolerance {_UPDATE_THRESHOLD_MM} mm); {growth}. The size this "
            f"figure saves at is therefore not reproducible — saving it again "
            f"may give a different one. Please report it at "
            f"https://github.com/jorgebotas/publiplots/issues, quoting this "
            f"message and the pp.subplots() / pp.legend() calls that built the "
            f"figure. To silence a case you know to be harmless: "
            f"warnings.filterwarnings('ignore', "
            f"category=pp.LayoutConvergenceWarning).",
            LayoutConvergenceWarning,
            stacklevel=2,
        )

    def _on_draw(self, event) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            new = self._measure()
            if self._needs_update(new):
                self._apply(new)
        finally:
            self._updating = False

    def _measure(self) -> Dict[str, Tuple[float, ...]]:
        fig = self._fig
        dpi = fig.dpi
        if dpi <= 0:
            return {}
        axes_matrix = self._axes_matrix()
        if not axes_matrix or not axes_matrix[0]:
            return {}

        managed = self._externally_managed_artist_ids()
        measured: Dict[str, Tuple[float, ...]] = {}

        for side, (axis_kind, calc) in _SIDE_CALCULATORS.items():
            if side in self._locked:
                continue
            locked_idxs = self._locked_positions.get(side, frozenset())
            current = getattr(self._layout, side)
            if axis_kind == "row":
                per = []
                for r, row in enumerate(axes_matrix):
                    if r in locked_idxs:
                        # Preserve the user-supplied locked value verbatim;
                        # tightbbox padding for an empty/blank-decoration
                        # row would otherwise re-inflate this slot every draw.
                        per.append(current[r])
                        continue
                    max_px = 0.0
                    for ax in row:
                        max_px = max(max_px, self._side_extent(ax, calc, managed))
                    per.append(max(max_px / dpi * 25.4, 0.0))
                measured[side] = tuple(per)
            else:  # "col"
                ncols = len(axes_matrix[0])
                per = []
                for c in range(ncols):
                    if c in locked_idxs:
                        per.append(current[c])
                        continue
                    max_px = 0.0
                    for row in axes_matrix:
                        ax = row[c]
                        max_px = max(max_px, self._side_extent(ax, calc, managed))
                    per.append(max(max_px / dpi * 25.4, 0.0))
                measured[side] = tuple(per)

        # Measure the single figure's legend_group (if any) and dispatch
        # its overhang to the right reservation field based on (side,
        # anchor_kind).
        self._apply_legend_band_measurement(measured, axes_matrix)
        # Measure pp.suptitle (if any) and write its mm height into
        # ``measured["suptitle_space"]``. Ordering is irrelevant —
        # suptitle_space is a dedicated scalar, independent of legend
        # bands.
        self._apply_suptitle_measurement(measured, dpi)
        return measured

    def _side_extent(self, ax, calc, managed_artist_ids) -> float:
        """Measure ax's tight-vs-window extent for one side, excluding managed overlays.

        Also accounts for reactor-managed artists that are NOT children of
        ``ax`` but are pinned to it — e.g. colorbar titles added via
        ``fig.text`` which live under the Figure, not the Axes.
        ``ax.get_tightbbox()`` misses those, so we manually union their
        window extents into the tight bbox before computing the side extent.
        """
        ax_bbox = ax.get_window_extent()
        # Temporarily drop managed overlay artists (legend_group's legends)
        # from layout consideration so they don't inflate the per-axis
        # reservations.
        toggled = []
        for child in ax.get_children():
            if id(child) in managed_artist_ids and child.get_in_layout():
                child.set_in_layout(False)
                toggled.append(child)
        try:
            tight = ax.get_tightbbox()
        finally:
            for child in toggled:
                child.set_in_layout(True)
        if tight is None:
            return 0.0

        # Union with pinned-but-not-child artists (per-axis colorbar
        # titles are fig.text artists registered to this axes via the
        # reactor). These sit inside the reserved side but are invisible
        # to ax.get_tightbbox() — without this union, the reservation
        # shrinks below what the title actually needs and the title gets
        # clipped on save.
        tight = self._union_pinned_artists(ax, tight, managed_artist_ids)
        return calc(ax_bbox, tight)

    def _union_pinned_artists(self, ax, tight, managed_artist_ids):
        """Union `tight` with extents of reactor-managed artists pinned to `ax`
        that are NOT among the excluded managed overlays."""
        reactor = getattr(self._fig, "_publiplots_layout_reactor", None)
        if reactor is None:
            return tight
        from matplotlib.transforms import Bbox
        for reg in reactor._registrations:
            if reg.ax is not ax:
                continue
            if id(reg.artist) in managed_artist_ids:
                continue  # external overlay — already excluded
            extent = self._artist_window_extent(reg.artist)
            if extent is None:
                continue
            tight = Bbox.union([tight, extent])
        return tight

    def _externally_managed_artist_ids(self) -> set:
        """IDs of LayoutReactor registrations flagged external_to_axis=True.

        The flag lives on _Registration in utils/layout_reactor.py (added by
        Task 4 of this amendment). Before Task 4 lands, getattr returns
        False and nothing is excluded — equivalent to pre-amendment
        behavior.
        """
        reactor = getattr(self._fig, "_publiplots_layout_reactor", None)
        if reactor is None:
            return set()
        return {
            id(reg.artist)
            for reg in reactor._registrations
            if getattr(reg, "external_to_axis", False)
        }

    _SCALAR_SIDES = {
        "legend_column", "legend_band_bottom", "legend_band_top", "legend_band_left",
        "suptitle_space",
    }

    def _needs_update(self, measured: Dict[str, Tuple[float, ...]]) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if side in self._SCALAR_SIDES:
                if abs(new_val - current) >= _UPDATE_THRESHOLD_MM:
                    return True
            else:
                # tuple comparison (per-row / per-col reservations)
                if len(new_val) != len(current):
                    return True
                for nv, cv in zip(new_val, current):
                    if abs(nv - cv) >= _UPDATE_THRESHOLD_MM:
                        return True
        return False

    # Per-side overhang calculators. Each returns the signed distance
    # (in pixels) that 'obj' projects past the 'anchor_bb' edge in the
    # chosen direction. Distances are non-negative after the max() below.
    _OVERHANG_BY_SIDE = {
        "right":  lambda anchor_bb, obj_bb: obj_bb.x1 - anchor_bb.x1,
        "left":   lambda anchor_bb, obj_bb: anchor_bb.x0 - obj_bb.x0,
        "bottom": lambda anchor_bb, obj_bb: anchor_bb.y0 - obj_bb.y0,
        "top":    lambda anchor_bb, obj_bb: obj_bb.y1 - anchor_bb.y1,
    }

    # side → (figure-anchored FigureLayout field, axes-anchored per-cell field)
    _FIELD_BY_SIDE = {
        "right":  ("legend_column",       "right",        "col"),
        "left":   ("legend_band_left",    "ylabel_space", "col"),
        "bottom": ("legend_band_bottom",  "xlabel_space", "row"),
        "top":    ("legend_band_top",     "title_space",  "row"),
    }

    def _apply_suptitle_measurement(self, measured: dict, dpi: float) -> None:
        """Measure the pixel height of ``fig._publiplots_suptitle`` (if any)
        and write it into ``measured["suptitle_space"]`` in mm.

        A ``+1 mm`` safety margin is added (same convention as
        :meth:`_measure_one_group` below at the
        ``overhang_mm = max_overhang_px / dpi * 25.4 + 1.0`` line).
        When there is no suptitle, writes ``0.0`` so the reservation
        collapses back to zero.
        """
        if "suptitle_space" in self._locked:
            return
        artist = getattr(self._fig, "_publiplots_suptitle", None)
        if artist is None:
            measured["suptitle_space"] = 0.0
            return
        try:
            extent = artist.get_window_extent()
        except Exception:
            measured["suptitle_space"] = 0.0
            return
        if extent is None or extent.height <= 0:
            measured["suptitle_space"] = 0.0
            return
        measured["suptitle_space"] = extent.height / dpi * 25.4 + 1.0

    def _apply_legend_band_measurement(self, measured: dict, axes_matrix) -> None:
        """Measure every pp.legend_group's overhang and write it into
        the correct FigureLayout reservation based on each group's
        ``side`` and anchor kind.

        Multiple groups may coexist on the same figure (each scoped via
        ``axes=``); each contributes its own measurement. Per-cell
        reservations accumulate via ``max()`` so two axes-anchored
        groups targeting different cells both get room.
        """
        groups = getattr(self._fig, "_publiplots_legend_groups", None)
        if not groups:
            return
        dpi = self._fig.dpi
        if dpi <= 0:
            return

        for group in groups:
            self._measure_one_group(group, measured, axes_matrix, dpi)

    def _measure_one_group(self, group, measured, axes_matrix, dpi) -> None:
        # Force materialization so artists exist to measure.
        group._materialize()
        if not group._builder.elements:
            return

        # Single-axes, in-frame groups (external_to_axis=False) are measured
        # by ax.get_tightbbox() in _side_extent — no overhang write needed
        # for the legend itself. This guard prevents double-counting against
        # the per-cell reservation when pp.legend(ax) routes through the same
        # group machinery.
        if not getattr(group, "_external_to_axis", True):
            # ... but a per-axes top legend must NOT sit between the axes
            # and its title (Issue B). The required stacking is
            # AXES -> LEGEND -> TITLE (title outermost). The legend renders
            # ~2mm above the axes top; we lift the title's pad above the
            # legend band so it clears it. The standard title_space
            # auto-measurement (which sees the lifted title at its new
            # position) then reserves room for both. _side_extent excludes
            # the legend artist itself, so it never double-counts.
            if group._anchor_kind == "axes":
                if group._side == "top":
                    self._lift_title_above_top_legend(group, dpi)
                elif group._side in ("left", "bottom"):
                    # A per-axes left or bottom legend must clear the tick
                    # labels / axis label that live on its own side of the
                    # axes rectangle, dynamically (Issue B for 'left';
                    # #212 for 'bottom') rather than at a fixed offset.
                    self._offset_inside_legend_past_decorations(
                        group, axes_matrix
                    )
            return

        side = group._side
        overhang_fn = self._OVERHANG_BY_SIDE[side]
        figure_field, cell_field, _ = self._FIELD_BY_SIDE[side]

        # Measure the overhang from the edge the band is actually placed
        # past — the outermost in-scope axes, not the anchor (#230).
        ref_axes, ref_idx = self._band_reference(group, axes_matrix)
        anchor_bb = self._reference_bbox(ref_axes)
        max_overhang_px = 0.0
        for _, obj in group._builder.elements:
            extent = self._artist_window_extent(obj)
            if extent is None:
                continue
            max_overhang_px = max(max_overhang_px, overhang_fn(anchor_bb, extent))
        if max_overhang_px <= 0:
            # Band doesn't overhang yet (first-draw before reactor has
            # repositioned). Still re-bake the decoration offset so a
            # group constructed BEFORE plots picks up the offset as soon
            # as its entries materialize.
            self._bake_decoration_offset(group, measured, ref_idx)
            return
        overhang_mm = max_overhang_px / dpi * 25.4 + 1.0

        if group._anchor_kind == "figure":
            if figure_field in self._locked:
                return
            # Figure-anchored: _GridAnchor already places the band past
            # all per-cell decorations, so no extra outward offset is
            # needed. Clear any stale offset from a prior layout pass.
            group._band_contribution_mm = overhang_mm
            group._set_decoration_offset(0.0)
            # Multiple figure-anchored groups on the same side compete
            # for the same band; take the tallest so neither clips.
            existing_scalar = measured.get(figure_field, 0.0)
            measured[figure_field] = max(existing_scalar, overhang_mm)
            return

        # Axes-anchored: grow the per-cell reservation for the outermost
        # in-scope row/column on this side. Merge with whatever the
        # auto-measurement already produced (so label/title space co-exists
        # with legend space).
        #
        # A pinned reservation forbids GROWING the row/column; it does not
        # forbid MOVING the band clear of the decorations (Issue #222). So
        # both lock guards below still skip the ``measured[cell_field]``
        # write, but they position the band first. The reservation-derived
        # ``pure_decoration_mm`` used further down is unusable here — under
        # a pin ``existing[idx]`` is the caller's mm, not a measurement — so
        # the offset is measured directly off ``ref_axes`` instead, and clamped
        # there to keep the band on the canvas (the pinned row/column will
        # not grow around it, and ``savefig.bbox`` is ``"standard"``).
        #
        # ``group._band_contribution_mm`` is deliberately NOT written on
        # either pinned branch: its only reader is
        # ``_bake_decoration_offset``, whose own guards are these same two
        # conditions on this same group and side, so nothing could ever read
        # it back.
        idx = ref_idx
        if cell_field in self._locked:
            self._offset_inside_legend_past_decorations(
                group, axes_matrix, ref_axes, idx
            )
            return
        if idx in self._locked_positions.get(cell_field, frozenset()):
            # Per-position lock: this slot is pinned (e.g., JointGrid's
            # joint↔marginal gap edges). Don't grow it for legend overhang.
            self._offset_inside_legend_past_decorations(
                group, axes_matrix, ref_axes, idx
            )
            return
        # Read the pure-decoration reservation BEFORE we write our
        # overhang into it. _measure() already filled ``measured[cell_field]``
        # from _side_extent (which uses set_in_layout(False) on managed
        # overlays, i.e., OUR legend, so the measured slot is the
        # anchor's decoration size WITHOUT our band).
        existing = list(measured.get(cell_field, getattr(self._layout, cell_field)))
        pure_decoration_mm = existing[idx]
        # For side='right' there is no decoration past ax.x1 (tick labels
        # live inside ax), so this is already 0. For side='top' it equals
        # the title height above ax.y1; for 'bottom' the xlabel+ticks
        # below ax.y0; for 'left' the ylabel+ticks left of ax.x0. The
        # band must step past that amount to avoid overlap.
        group._band_contribution_mm = overhang_mm
        group._set_decoration_offset(pure_decoration_mm)
        # Grow the reservation by overhang_mm (capped against prior value
        # so multiple overlapping groups don't shrink it).
        existing[idx] = max(existing[idx], overhang_mm + pure_decoration_mm)
        measured[cell_field] = tuple(existing)

    def _band_reference(self, group, axes_matrix):
        """Axes the band is actually placed past, and their grid index.

        ``LegendBuilder`` anchors a band to ``_anchor_ax``: the anchor axes
        for a single-axes scope, but the scope's ``_ScopeAnchor`` (i.e. the
        union rect of every in-scope axes) for a multi-axes one. The reactor
        therefore places the band past the OUTERMOST in-scope axes on the
        band's side, which is not the anchor whenever the caller anchored to
        an inner axes — ``pp.legend(anchor=axes[0], axes=[axes[0], axes[1]],
        side='right')``.

        Measuring the overhang from the anchor there, and writing it into the
        ANCHOR's row/column, made the layout non-convergent (#230): the
        overhang then spans every intervening axes plus the gaps, and the
        reservation that grows is one of the cells that pushes those axes —
        and with them the band — further out, so the next pass measures more
        again, without limit. Anchoring both halves to the outermost in-scope
        cell closes the loop, for two different reasons depending on the side:

        - On 'right' / 'top' the reservation (``right[idx]`` /
          ``title_space[idx]``) is appended OUTSIDE the cell's outer edge, so
          growing it cannot move the edge the overhang is measured from.
        - On 'left' / 'bottom' it can: ``ylabel_space[idx]`` /
          ``xlabel_space[idx]`` sit inside the cell's origin, so growing them
          shifts the reference cell. But it shifts the whole in-scope block
          rigidly, band included, and the overhang is a RELATIVE distance
          (``anchor_bb.x0 - obj_bb.x0``), which a rigid translation leaves
          unchanged.

        Either way the measurement is invariant under the resize it causes, so
        the layout settles in one pass instead of advancing forever.

        Returns ``(axes_list, index)``. ``index`` is the row index for
        ``side`` in ``('top', 'bottom')`` and the column index otherwise, and
        is ``None`` for a figure-anchored group — that path writes a
        figure-level scalar, not a per-cell reservation, so it has no index
        and ``axes_list`` is the ``_GridAnchor`` proxy itself.
        """
        ref = getattr(group._builder, "_anchor_ax", None)
        if ref is None:
            ref = group.anchor
        scope_fn = getattr(ref, "scope_axes", None)
        scope = list(scope_fn()) if callable(scope_fn) else [ref]
        if not scope:
            scope = [group.anchor]
        side = group._side
        axis_kind = self._FIELD_BY_SIDE[side][2]
        if group._anchor_kind != "axes":
            return scope, None

        rows, cols = self._find_scope_indices(scope, axes_matrix)
        idxs = cols if axis_kind == "col" else rows
        if not idxs:
            # No scope axes resolve to a grid cell. Twins DO resolve (see
            # _find_scope_indices / _cell_siblings), so what reaches here is
            # an axes with no cell of its own — an ``ax.inset_axes`` child, or
            # a figure publiplots did not build — and we fall back to the
            # anchor.
            #
            # For a SINGLE-axes scope that is exact: anchor and outermost
            # axes coincide, which is the whole reason the single-axes case
            # was always immune to #230. For a MULTI-axes scope it is a best
            # effort, NOT a fix: the band is still placed past the scope's
            # union while the reservation grows the anchor's cell, so the
            # #230 feedback loop survives there unless the anchor happens to
            # be the outermost member. Closing it needs those axes to resolve
            # to cells, as twins now do. (``_find_ax_indices`` also answers
            # (0, 0) for an anchor that is not in the grid at all, which is
            # why that is a fallback and not a result.)
            r, c = self._find_ax_indices(group.anchor, axes_matrix)
            return [group.anchor], (c if axis_kind == "col" else r)
        # Row 0 is the TOP row (FigureLayout.axes_position stacks upward from
        # the last row), so 'top' takes the first row and 'bottom' the last.
        idx = idxs[-1] if side in ("right", "bottom") else idxs[0]
        # Alias ids, so a twin in the scope selects the GRID axes sharing its
        # cell — the reference must be an axes the layout owns: it carries the
        # cell's rectangle and is what _offset_inside_legend_past_decorations
        # takes a tight bbox of.
        scope_ids = self._cell_alias_ids(scope)
        cells = (
            [row[idx] for row in axes_matrix] if axis_kind == "col"
            else list(axes_matrix[idx])
        )
        outer = [a for a in cells if id(a) in scope_ids]
        return (outer or [group.anchor]), idx

    def _reference_bbox(self, ref_axes):
        """Pixel bbox of the reference axes, unioned across the outer cell.

        For the band's own side the union picks exactly the edge the reactor
        anchors to (max ``x1`` for 'right', min ``x0`` for 'left', and so on),
        and collapses to the single axes' extent for a single-axes scope.
        """
        from matplotlib.transforms import Bbox
        return Bbox.union([ax.get_window_extent() for ax in ref_axes])

    def _lift_title_above_top_legend(self, group, dpi) -> None:
        """Push the anchor axes' title pad above a per-axes top legend band.

        Issue B: a per-axes ``side='top'`` legend renders ~``x_offset`` mm
        above the axes top. matplotlib ignores it when auto-positioning the
        title (the legend is ``set_in_layout(False)``), so by default the
        title lands ~6pt above the axes — sandwiched *under* the legend.

        We compute the title pad needed to clear the whole legend band
        (outward gap + legend height + a small breathing gap) and apply it
        via ``ax.title`` offset. The standard ``title_space`` auto-measure
        then sees the title at its lifted position and reserves room for
        the legend + title together — no manual reservation arithmetic and
        no double-counting (``_side_extent`` excludes the legend artist).

        Convergent by construction: the pad is derived only from geometry
        that does NOT move between convergence iterations. Each element
        contributes ``outward_mm + (its top - its own placement reference)``
        — a *relative* rise, so it is independent of where the band
        currently sits — and ``outward_mm`` is read straight off the
        element's reactor registration, i.e. the distance the reactor
        *will* place it at. Re-running therefore recomputes the same pad.

        Measuring the band's absolute top against ``ax.y1`` instead would
        bake in a mid-convergence position: on the first draw the band has
        not been repositioned by the reactor yet, so it can read several mm
        high and leave that much stale padding behind (the lift only ever
        grows the pad, so nothing later takes it back out). Spanning the
        band's own ``min(y0) → max(y1)`` instead is wrong for a different
        reason: an element that extends *below* its placement reference
        would have that overhang counted a second time on top of
        ``outward_mm``.
        """
        ax = group.anchor
        title = getattr(ax, "title", None)
        if title is None or not title.get_text():
            return  # no title to lift

        # Reactor registration per element: it carries the outward distance
        # (mm from the axes' top edge) the element is placed at, which is
        # stable, unlike the element's current pixel position.
        regs = {
            id(reg.artist): reg
            for reg in group._builder._reactor._registrations
        }
        band_above_mm = 0.0
        for _, obj in group._builder.elements:
            # Tight bbox for the TOP: it includes decorations that sit past
            # the artist's rectangle (colorbar tick labels / titles).
            extent = self._artist_window_extent(obj)
            reg = regs.get(id(obj))
            if extent is None or reg is None:
                continue
            # Placement reference for the BOTTOM. For side='top' the reactor
            # anchors a colorbar by its strip rectangle's bottom edge
            # (layout_reactor: ``bottom_y = new_y``), which get_tightbbox
            # undershoots by the tick labels below the strip; Legend/Text
            # have no separate rectangle, so their own extent is the
            # reference (Legend is anchored by its bottom too).
            base = extent
            obj_ax = getattr(obj, "ax", None)
            if obj_ax is not None and hasattr(obj_ax, "get_window_extent"):
                base = obj_ax.get_window_extent()
            if reg.mm_height is not None:
                # A colorbar strip is added in figure *fractions*, so its
                # pixel height lags one resize behind whenever this pass
                # grows the figure — the reactor only restores the declared
                # mm at the end of the draw. Take the height from the
                # registration (authoritative, and what the reactor will
                # apply) and measure only the decoration overhang above the
                # strip, which is text and doesn't scale with the figure.
                rise_mm = reg.mm_height + (extent.y1 - base.y1) / dpi * 25.4
            else:
                rise_mm = (extent.y1 - base.y0) / dpi * 25.4
            outward_mm = (
                reg.mm_x_from_right + reg.mm_outward_decoration_offset
            )
            band_above_mm = max(band_above_mm, outward_mm + rise_mm)
        if band_above_mm <= 0:
            return  # band hasn't rendered yet
        band_above_ax_px = band_above_mm * dpi / 25.4

        # pad (points) = band height above axes + a small breathing gap.
        # matplotlib's title pad positions the title baseline; the text's
        # own descent adds visible space below it, so a tiny nominal gap
        # keeps the legend->title gap near ~2mm rather than ballooning.
        breathing_mm = 1.0
        breathing_px = breathing_mm * dpi / 25.4
        pad_px = band_above_ax_px + breathing_px
        pad_pt = pad_px / dpi * 72.0
        # Preserve the user's intended ("base") title pad: stash it once,
        # BEFORE we ever lift, so re-running over convergence iterations
        # doesn't read our own lifted pad as the baseline (idempotent — the
        # applied pad converges instead of drifting). Default to
        # rcParams['axes.titlepad'] when the user never set one.
        base_pad = getattr(ax, "_pp_base_titlepad", None)
        if base_pad is None:
            cur = ax.titleOffsetTrans._t[1] * 72.0  # current pad in points
            base_pad = cur if cur > 0 else float(
                mpl.rcParams.get("axes.titlepad", 6.0)
            )
            ax._pp_base_titlepad = base_pad
        # Only lift; honour a user-set pad larger than the band lift.
        # Apply via the title offset transform so the title's own
        # font/color/weight styling is NOT reset (ax.set_title would
        # re-merge the default fontdict and clobber user styling).
        ax._set_title_offset_trans(max(base_pad, pad_pt))

    def _offset_inside_legend_past_decorations(
        self, group, axes_matrix, ref_axes=None, ref_idx=None
    ) -> None:
        """Step a per-axes left/bottom legend just past the tick labels and
        axis label on its own side, as far as the canvas allows.

        Issue B (``side='left'``): with the old fixed 8mm offset removed, an
        internal left legend would render only ``x_offset`` mm left of the
        axes spine, colliding with the y-tick labels (which live left of the
        axes). Issue #212 (``side='bottom'``): identically, a bottom band was
        placed a fixed ``x_offset`` mm below the axes rect without measuring
        the x-tick labels / xlabel already occupying that space, overlapping
        them by ~1.6mm on a 50x40mm axes. Both sides — categorical legends
        and colorbar bands alike — are cured by the same measurement, so they
        share this one implementation; only the side's field and axis differ,
        which ``_FIELD_BY_SIDE`` / ``_OVERHANG_BY_SIDE`` already encode.

        For an in-frame (``external_to_axis=False``) group only 'left' and
        'bottom' route here: ``side='right'`` needs nothing (no decoration
        lives past ``ax.x1``) and ``side='top'`` is handled by
        ``_lift_title_above_top_legend`` instead, because there the title
        must end up OUTSIDE the band, so the title moves rather than the
        band. The pinned axes-anchored path in ``_measure_one_group`` also
        calls this for the remaining sides (see below); the measurement is
        side-generic — ``_OVERHANG_BY_SIDE`` supplies the direction — and
        collapses to ~0 on a side with no decoration.

        Unlike a figure-anchored band, an ``external_to_axis=False`` group
        is NOT excluded from ``ax.get_tightbbox()``, so the standard
        ``ylabel_space`` / ``xlabel_space`` auto-measurement ALREADY grows the
        column/row to fit the legend — we must not re-add the legend size
        (that double-counts and drifts). All we need is to position the legend
        just past the PURE decoration (ticklabels + axis label) so it doesn't
        overlap it. We measure that pure extent directly here (excluding our
        own legend) and bake it as the band's outward decoration offset.
        Idempotent, and it collapses to 0 on an axes with no tick labels and
        no axis label — no fixed gap is ever added.

        Deliberately NOT gated on ``self._locked`` /
        ``self._locked_positions`` (Issue #222). A pinned ``xlabel_space`` /
        ``ylabel_space`` means "do not GROW this reservation", which is what
        the guards in ``_measure_one_group`` enforce — that path writes
        ``measured[cell_field]``. This method writes nothing but the band's
        outward *position*, so gating it here conflated the pin with "do not
        MOVE the band clear of the decorations" and dropped the band on top
        of the tick labels and axis label.

        The lock sets are consulted for one thing only: whether the step
        outward has to be CLAMPED to keep the band on the canvas. When the
        reservation auto-measures, the row/column grows to contain the band
        (an ``external_to_axis=False`` band is inside ``ax.get_tightbbox()``,
        and an external one is added as an overhang), so the band can never
        leave the figure and there is nothing to clamp — clamping there would
        actively harm, because mid-convergence the figure is transiently too
        small and the reservation would then settle around the clamped
        position, freezing the band short of the decorations. A pinned
        reservation is exactly the case where the figure will NOT grow, so
        the clamp applies and the priority order is:

        1. the band stays fully inside the figure — ``savefig.bbox`` is
           ``"standard"``, so anything outside is cropped out of the saved
           file, and a deleted legend is far worse than an overlapping one;
        2. subject to that, step as far past the decorations as fits, so any
           residual overlap is the minimum achievable rather than the full
           pre-#222 overlap.

        The floor is 0 mm — the offset a pinned reservation got before #222 —
        so a pin too small to fit even the band alone degrades exactly to the
        old placement and never moves the band further INWARD than that.

        ``ref_axes`` / ``ref_idx`` come from :meth:`_band_reference`: the
        outermost in-scope axes on this side, which is the edge the reactor
        actually places the band past (#230). They are computed here when the
        caller does not supply them. For a single-axes scope — every
        ``external_to_axis=False`` group — they collapse to the anchor and its
        own cell, i.e. the pre-#230 behaviour exactly.
        """
        side = group._side
        dpi = self._fig.dpi
        if ref_axes is None or ref_idx is None:
            ref_axes, ref_idx = self._band_reference(group, axes_matrix)
        ax_bb = self._reference_bbox(ref_axes)

        # Pure decoration extent past the axes edge on this side, EXCLUDING
        # our legend (and any other externally-managed overlay). Mirrors
        # _side_extent but computed locally so it's independent of whether
        # the legend is in-layout. Taken as the max over the outer cell's
        # in-scope axes, matching how _measure fills a per-cell reservation.
        legend_ids = {id(obj) for _, obj in group._builder.elements}
        # Exclude our own legend from BOTH the tightbbox and the pinned
        # union. The group is external_to_axis=False, so its legend is NOT
        # in _externally_managed_artist_ids() and would otherwise be unioned
        # back in by _union_pinned_artists — re-inflating the "pure" extent
        # by the legend's own width and causing the offset to drift.
        managed = self._externally_managed_artist_ids() | legend_ids
        pure_decoration_mm = None
        for ax in ref_axes:
            toggled = []
            for child in ax.get_children():
                if id(child) in legend_ids and child.get_in_layout():
                    child.set_in_layout(False)
                    toggled.append(child)
            try:
                tight = ax.get_tightbbox()
            finally:
                for child in toggled:
                    child.set_in_layout(True)
            if tight is None:
                continue
            tight = self._union_pinned_artists(ax, tight, managed)
            one = (
                self._OVERHANG_BY_SIDE[side](ax.get_window_extent(), tight)
                / dpi * 25.4
            )
            pure_decoration_mm = (
                one if pure_decoration_mm is None
                else max(pure_decoration_mm, one)
            )
        if pure_decoration_mm is None:
            return

        offset_mm = pure_decoration_mm
        if self._is_pinned_cell(side, ref_idx):
            offset_mm = min(
                offset_mm, self._max_onscreen_offset_mm(group, side, ax_bb, dpi)
            )
        # Single floor for both paths: a side whose tight bbox does not reach
        # past the axes edge measures a negative "decoration", and a pin too
        # small for the band alone makes the clamp negative. Either way the
        # band must not be pulled INWARD of the pre-#222 placement.
        group._set_decoration_offset(max(0.0, offset_mm))

    def _is_pinned_cell(self, side, idx) -> bool:
        """True when the reservation the band draws into is user-pinned.

        ``idx`` is the row/column the band's overhang would grow — the
        outermost in-scope cell on this side (:meth:`_band_reference`).
        Whole-side (``self._locked``) and per-position
        (``self._locked_positions``) pins both count — in either case
        ``_measure`` will not grow that row/column, so the figure cannot
        stretch to contain a band stepped outward past the decorations.
        """
        _, cell_field, _ = self._FIELD_BY_SIDE[side]
        if cell_field in self._locked:
            return True
        return idx in self._locked_positions.get(cell_field, frozenset())

    def _max_onscreen_offset_mm(self, group, side, ax_bb, dpi) -> float:
        """Largest outward offset (mm) that keeps the whole band on the canvas.

        The reactor places each element at ``mm_x_from_right +
        mm_outward_decoration_offset`` from the axes edge and the element then
        extends its own size further outward, so the band's total reach is
        ``base_gap + offset + own_size`` and the offset that just touches the
        figure edge is ``available - base_gap - own_size``.

        ``available`` is measured, not derived from ``FigureLayout``: the
        pixel gap between the anchor's edge and the figure edge already
        accounts for ``outer_pad``, every reservation stacked between them and
        (for a non-edge row/column) the neighbouring cells, none of which this
        method would otherwise have to re-derive. ``own_size`` is taken from
        the element's own tight bbox rather than from its current position, so
        it does not go stale when ``_apply`` has repositioned the axes but the
        reactor has not yet re-anchored the artists.

        Returns ``+inf`` when nothing measurable exists, so the caller's
        ``min()`` becomes a no-op.
        """
        fig_bb = self._fig.get_window_extent()
        if side == "bottom":
            available_mm = (ax_bb.y0 - fig_bb.y0) / dpi * 25.4
        elif side == "top":
            available_mm = (fig_bb.y1 - ax_bb.y1) / dpi * 25.4
        elif side == "left":
            available_mm = (ax_bb.x0 - fig_bb.x0) / dpi * 25.4
        else:  # "right"
            available_mm = (fig_bb.x1 - ax_bb.x1) / dpi * 25.4

        regs = {
            id(reg.artist): reg
            for reg in group._builder._reactor._registrations
        }
        vertical = side in ("bottom", "top")
        band_mm = 0.0
        for _, obj in group._builder.elements:
            extent = self._artist_window_extent(obj)
            reg = regs.get(id(obj))
            if extent is None or reg is None:
                continue
            size_mm = (extent.height if vertical else extent.width) / dpi * 25.4
            # A colorbar strip is sized in figure fractions, so its pixel
            # size lags one resize behind whenever the layout is growing.
            # The declared mm is authoritative and is what the reactor will
            # restore, so use it as a floor on the outward dimension.
            declared = reg.mm_height if vertical else reg.mm_width
            if declared is not None:
                size_mm = max(size_mm, declared)
            band_mm = max(band_mm, reg.mm_x_from_right + size_mm)
        if band_mm <= 0.0:
            return float("inf")
        return available_mm - band_mm

    def _bake_decoration_offset(self, group, measured, ref_idx) -> None:
        """Write the decoration offset onto the group's registrations
        without touching its reservation. Used on first draw when the
        band hasn't rendered yet (no overhang to measure) but we still
        want the band to land past decorations once it materializes.

        The lock guards below are KEPT (unlike the ones removed from
        ``_offset_inside_legend_past_decorations`` for Issue #222), because
        here the offset is *derived from the reservation*:
        ``existing[idx] - group._band_contribution_mm``. That subtraction is
        only a valid stand-in for the pure decoration extent while
        ``existing[idx]`` is an auto-measurement. Under a pin it is the
        caller's pinned mm, so the difference is arbitrary — a generous pin
        would fling the band far outward, a tight one would clamp it to 0.
        Skipping the pre-bake is harmless: this is a first-draw estimate for
        a band that has not rendered yet, and the very next pass (once the
        band has an overhang) supersedes it with a real measurement —
        ``_measure_one_group``'s axes-anchored path for
        ``external_to_axis=True`` groups, and
        ``_offset_inside_legend_past_decorations``, which measures the pure
        decoration directly, for ``external_to_axis=False`` ones.

        ``ref_idx`` is the cell :meth:`_band_reference` resolved for this
        group — the outermost in-scope row/column on the band's side, i.e.
        the same slot the non-first-draw path grows (#230).

        .. warning::

           This method appears to be DEAD CODE, and the ``ref_idx`` change is
           therefore theoretically right but empirically inert. Instrumenting
           it across the full 2017-test suite recorded ZERO calls, and an
           independent review could not reach it in 14 hand-built
           configurations — including the "group constructed BEFORE the plot
           calls" ordering the paragraph above names as its reason to exist,
           and including a pinned cell.
           Its guard is ``max_overhang_px <= 0``, and by the time
           ``_measure_one_group`` runs, ``group._materialize()`` has already
           created the artists and the reactor has already placed them past
           the anchor edge, so the overhang is positive on the very first
           pass. Left in place rather than deleted because proving a negative
           over every entry point is not the same as proving it unreachable;
           treat any change here as unverifiable by test.
        """
        if group._anchor_kind != "axes" or ref_idx is None:
            return
        side = group._side
        _, cell_field, _ = self._FIELD_BY_SIDE[side]
        if cell_field in self._locked:
            return
        existing = measured.get(cell_field, getattr(self._layout, cell_field))
        idx = ref_idx
        if idx in self._locked_positions.get(cell_field, frozenset()):
            return
        pure_decoration_mm = existing[idx] - group._band_contribution_mm
        if pure_decoration_mm < 0:
            pure_decoration_mm = 0.0
        group._set_decoration_offset(pure_decoration_mm)

    def _find_ax_indices(self, ax, axes_matrix):
        for r, row in enumerate(axes_matrix):
            for c, a in enumerate(row):
                if a is ax:
                    return r, c
        return 0, 0

    @staticmethod
    def _cell_siblings(ax):
        """``ax`` plus every axes matplotlib considers twinned with it.

        ``ax.twinx()`` / ``ax.twiny()`` build a second Axes that occupies
        EXACTLY the parent's cell — its ``axes_locator`` is pinned to the
        parent's ``transAxes``, so it tracks the parent through every
        ``_apply`` repositioning — but that twin is not in
        ``fig._publiplots_axes`` and so is invisible to an ``is`` comparison
        against the grid. Resolving it to its parent's cell is what lets
        ``_band_reference`` treat a twin scope like any other (#230).

        The twinned-axes grouper is the key, NOT ``get_subplotspec()``:
        ``pp.subplots`` builds every cell with ``fig.add_axes``, so parent and
        twin both report ``None`` there and matching on it would collapse the
        entire grid into a single bucket. ``_twinned_axes`` is private
        matplotlib API, hence the defensive getattr; it degrades to identity
        matching, i.e. the behaviour before this helper existed.

        Returns ``[ax]`` for an axes with no twin.
        """
        grouper = getattr(ax, "_twinned_axes", None)
        if grouper is None:
            return [ax]
        try:
            siblings = list(grouper.get_siblings(ax))
        except Exception:
            return [ax]
        return siblings or [ax]

    def _cell_alias_ids(self, scope_axes) -> set:
        """ids of every axes that resolves to the same grid cell as one of these."""
        ids = set()
        for ax in scope_axes:
            ids.update(id(a) for a in self._cell_siblings(ax))
        return ids

    def _find_scope_indices(self, scope_axes, axes_matrix):
        """Return (row_indices, col_indices) touched by any axes in scope_axes.

        scope_axes is a list of matplotlib Axes. Returns two sorted lists of
        unique row/col indices within axes_matrix. Used by commit 4's
        multi-axes scope path to aggregate per-cell reservations via max().

        Matching is by grid CELL, not by object identity: an axes that shares
        a cell with a grid axes (a twin — see :meth:`_cell_siblings`) resolves
        to that cell, so a band scoped over twins reserves space in the
        columns/rows those twins are drawn in.
        """
        alias_ids = self._cell_alias_ids(scope_axes)
        rows, cols = set(), set()
        for r, row in enumerate(axes_matrix):
            for c, a in enumerate(row):
                if id(a) in alias_ids:
                    rows.add(r)
                    cols.add(c)
        return sorted(rows), sorted(cols)

    def _artist_window_extent(self, obj):
        """Duck-typed tight-bbox accessor (Legend/Colorbar/Text).

        Returns the *tight* bbox, not the bare window extent — the tight
        bbox includes decorations attached outside the artist's
        rectangle, like colorbar tick labels and titles sitting past the
        color strip. Without that, the reactor measures only the narrow
        color strip and the tick labels get clipped on save.

        Legend.get_window_extent() already equals its tightbbox (the
        legend packs its own frame internally), so this call is
        idempotent there.
        """
        # Colorbar-like: geometry lives on a child Axes, which exposes
        # get_tightbbox. Use it so tick labels / titles are included.
        if hasattr(obj, "ax") and hasattr(obj.ax, "get_tightbbox"):
            return obj.ax.get_tightbbox()
        if hasattr(obj, "get_tightbbox"):
            try:
                return obj.get_tightbbox()
            except TypeError:
                # Some artists require a renderer arg.
                pass
        if hasattr(obj, "get_window_extent"):
            return obj.get_window_extent()
        if hasattr(obj, "ax"):
            return obj.ax.get_window_extent()
        return None

    def _apply(self, measured: Dict[str, Tuple[float, ...]]) -> None:
        new_layout = self._layout.with_updated_reservations(**measured)
        self._layout = new_layout
        self._fig._publiplots_layout = new_layout

        W, H = new_layout.figure_size()
        # forward=True propagates the new size to the GUI canvas so plt.show()
        # renders at the resized dimensions. Without it, the canvas keeps its
        # initial size and decorations that grew into the extra reservation get
        # cropped. The re-entrance guard (_updating) plus the 0.1 mm threshold
        # in _needs_update() keep this loop-safe.
        self._fig.set_size_inches(W * _MM2INCH, H * _MM2INCH, forward=True)

        for r, row in enumerate(self._axes_matrix()):
            for c, ax in enumerate(row):
                ax.set_position(new_layout.axes_position(r, c))

        # Reposition pp.suptitle (if any) to the vertical midpoint of
        # its reserved band. Runs after set_size_inches so the figure's
        # final height is known; the next draw renders the title at
        # the fixed fraction inside the grown canvas.
        suptitle = getattr(self._fig, "_publiplots_suptitle", None)
        if suptitle is not None and new_layout.suptitle_space > 0:
            y_mm = H - new_layout.outer_pad - new_layout.suptitle_space / 2
            suptitle.set_position((0.5, y_mm / H))
            suptitle.set_verticalalignment("center")
            suptitle.set_horizontalalignment("center")

    def _axes_matrix(self):
        stored = getattr(self._fig, "_publiplots_axes", None)
        if stored is not None:
            return stored
        flat = list(self._fig.axes)
        nrows, ncols = self._layout.nrows, self._layout.ncols
        if len(flat) < nrows * ncols:
            return [[]]
        return [flat[r * ncols:(r + 1) * ncols] for r in range(nrows)]
