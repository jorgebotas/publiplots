"""
Shared legend band across multiple subplots.

MultiAxesLegendGroup composes one unified legend band anchored to a
chosen side of the figure (or a single axes) even when individual
legends/colorbars are attached to other axes in the same figure. This
is the primary tool for complex subplot layouts.
"""

import warnings
from typing import List, Optional, Sequence

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
    """

    def __init__(self, fig: Figure) -> None:
        self._fig = fig

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
                y1 = (H - layout.outer_pad - layout.legend_band_top) / H
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
    ):
        if side not in ("right", "left", "bottom", "top"):
            raise ValueError(
                f"side must be 'right' | 'left' | 'bottom' | 'top', got {side!r}"
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
        self._orientation = (
            self._DEFAULT_ORIENTATION[side] if orientation == "auto" else orientation
        )
        self._align = (
            self._DEFAULT_ALIGN[side] if align == "auto" else align
        )
        if x_offset is None:
            x_offset = self._DEFAULT_X_OFFSET_MM[side]

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
        # anchor_ax=self.anchor pins position math/reactor to the anchor
        # regardless of self.ax swaps during add_* calls. For
        # figure-anchored groups anchor_ax is the _GridAnchor proxy,
        # which exposes the same get_position()/get_figure() API.
        builder_ax = self.anchor if self._anchor_kind == "axes" \
                     else self._pick_builder_ax()
        self._builder = LegendBuilder(
            ax=builder_ax,
            anchor_ax=self.anchor,
            x_offset=x_offset,
            y_offset=y_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            max_width=max_width,
            external_to_axis=True,
            side=side,
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
        """True if ``self`` and ``other`` share any axes."""
        # Full-grid scope overlaps everything.
        if self._scope_axes is None or other._scope_axes is None:
            return True
        self_ids = {id(a) for a in self._scope_axes}
        return any(id(a) in self_ids for a in other._scope_axes)

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
        """
        target_ax = ax if ax is not None else self._default_target_ax()
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


def legend_group(
    anchor: Optional[Axes] = None,
    *,
    side: str = "right",
    figure: Optional[Figure] = None,
    orientation: str = "auto",
    align: str = "auto",
    collect: Optional[Sequence[str]] = None,
    axes: Optional[Sequence[Axes]] = None,
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: Optional[float] = None,
    max_width: Optional[float] = None,
) -> MultiAxesLegendGroup:
    """Create a shared legend band.

    Pass ``anchor=None`` (default) for a **figure-anchored** band that
    spans the full subplot grid on the chosen side. Pass
    ``anchor=axes[r, c]`` to pin the band to a single cell; the
    corresponding per-cell reservation (``right[c]`` for ``side='right'``,
    ``xlabel_space[r]`` for ``side='bottom'``, etc.) absorbs the band
    width.

    ``axes=`` scopes collection to a subset of the grid — only entries
    stashed on those axes are gathered, and only per-axis legends on
    those axes are evicted. Multiple groups on one figure (each with a
    disjoint ``axes=``) let you render independent bands for different
    subplots.

    ``orientation`` defaults to ``'horizontal'`` for ``side='top'|'bottom'``
    (entries run along the edge left-to-right, multiple legends sit
    side-by-side, overflow spills outward in new rows) and
    ``'vertical'`` for ``'right'|'left'`` (current behaviour).
    ``align`` defaults to ``'center'`` for top/bottom and ``'start'``
    for left/right.

    See :class:`MultiAxesLegendGroup` for all parameter docs.
    """
    return MultiAxesLegendGroup(
        anchor=anchor,
        side=side,
        figure=figure,
        orientation=orientation,
        align=align,
        collect=collect,
        axes=axes,
        x_offset=x_offset,
        y_offset=y_offset,
        gap=gap,
        column_spacing=column_spacing,
        vpad=vpad,
        max_width=max_width,
    )