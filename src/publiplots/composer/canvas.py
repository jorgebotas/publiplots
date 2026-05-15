"""Canvas — the Composer's top-level orchestrator.

Supports multi-row layouts via repeated :meth:`Canvas.add_row` calls
(PR 3) and raster :meth:`Canvas.savefig`. Figure creation is LAZY:
``add_row`` only stages a row record; the matplotlib :class:`Figure`
is materialized on first access of ``canvas.figure``,
``canvas[label]``/``canvas[i]``, or ``canvas.savefig`` (or via an
explicit :meth:`Canvas.finalize` call).

Internally, the canvas builds an N-row × C-column geometry
(via :func:`compute_canvas_geometry`) where each row has its own
panel widths and a per-row ``vpad`` (mm above the row).

For single-row canvases, a :class:`SubplotsAutoLayout` reactor is
attached so axis decorations (xlabel/ylabel/title) auto-grow on first
draw to fit decorations. Multi-row canvases skip the reactor and use
static rcParams reservations (refined in PR 4.5).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from publiplots.composer.exceptions import ComposerOverflowError
from publiplots.composer.panels import Panel, PanelAxes, PanelImage
from publiplots.composer.presets import resolve_preset
from publiplots.themes.rcparams import resolve_param


def _panel_raw_width(panel) -> Any:
    """Return the panel's raw width input for flex resolution.

    - PanelAxes / PanelText / PanelImage: ``panel.size[0]`` (float or ``'flex'`` sentinel)
    - PanelGrid: ``panel.size_mm[0]`` (computed grid outer width; never flex)

    PanelGrid has no ``size`` attribute — its outer width is computed
    from ``shape`` + ``axes_size`` + spacing via the ``size_mm`` property.
    Reading ``panel.size[0]`` unconditionally would AttributeError on
    PanelGrid inputs to :meth:`Canvas.add_row`.
    """
    from publiplots.composer.panels import PanelGrid
    if isinstance(panel, PanelGrid):
        return panel.size_mm[0]
    # PanelAxes, PanelText, PanelImage all expose size: (w_or_flex, h)
    return panel.size[0]


def _panel_raw_height(panel) -> float:
    """Return the panel's height in mm.

    - PanelAxes / PanelText / PanelImage: ``panel.size[1]``
    - PanelGrid: ``panel.size_mm[1]`` (computed grid outer height)
    """
    from publiplots.composer.panels import PanelGrid
    if isinstance(panel, PanelGrid):
        return panel.size_mm[1]
    return panel.size[1]


@dataclass
class _RowStaging:
    """Internal record of an :meth:`Canvas.add_row` call.

    Attributes
    ----------
    panels : tuple of PanelAxes (or PanelGrid/PanelText in PR 3 Tasks 7+)
        Panel input records as supplied by the user.
    vpad_mm : float
        Vertical pad in mm above this row. Set to 0.0 for the first
        row regardless of the user-supplied value (avoids extra top
        padding); user-supplied for subsequent rows.
    """

    panels: tuple
    vpad_mm: float


class Canvas:
    """Programmatic mm-precise canvas for multi-panel paper figures.

    Parameters
    ----------
    preset : str
        Preset name. Supports the 12 journal presets (Cell, Nature,
        Nature Methods, Science families) plus ``'custom'`` as an
        escape hatch for arbitrary widths.
    width : float, optional
        Canvas width in millimeters. Required for ``preset='custom'``.

    Examples
    --------
    Single-row, two axes panels:

    >>> import publiplots as pp
    >>> canvas = pp.Canvas("custom", width=174.0)
    >>> canvas.add_row(
    ...     pp.PanelAxes(label="A", size=(70, 40)),
    ...     pp.PanelAxes(label="B", size=(70, 40)),
    ... )
    >>> pp.scatterplot(data=df, x="x", y="y", ax=canvas["A"].ax)
    >>> canvas.savefig("fig.png")

    Multi-row layouts (PR 3):

    >>> canvas = pp.Canvas("custom", width=174.0)
    >>> canvas.add_row(pp.PanelAxes(label="A", size=(70, 40)),
    ...                pp.PanelAxes(label="B", size=(70, 40)))
    >>> canvas.add_row(pp.PanelAxes(label="C", size=(140, 30)))
    >>> canvas.savefig("fig.png")  # triggers figure finalization

    Notes
    -----
    Vector PDF (PR 5) and vector SVG (PR 6a) save dispatches are
    wired. ``canvas.savefig('fig.pdf')`` and ``canvas.savefig('fig.svg')``
    composite any PanelImage slots into the output.
    """

    def __init__(
        self,
        preset: str,
        *,
        width: Optional[float] = None,
        abc: Union[str, bool, Sequence[str], None] = None,
        strict_vectors: bool = False,
    ) -> None:
        """Initialize the canvas.

        Parameters
        ----------
        preset : str
            Preset name (12 journal presets or ``'custom'``).
        width : float, optional
            Canvas width in mm. Required for ``preset='custom'``.
        abc : str, bool, or sequence, optional
            abc auto-letter mode override.
        strict_vectors : bool, default False
            When True, ``canvas.savefig('fig.pdf')`` raises
            :class:`ComposerVectorError` if any PanelImage schematic
            fails to load as vector (corrupt SVG, missing optional dep,
            malformed PDF). When False (default), failed vector loads
            fall back to a raster re-render of the schematic and emit a
            ``UserWarning``.
        """
        from publiplots.composer.abc_labels import DEFAULT_LABEL_STYLE

        self._preset_name = preset
        spec = resolve_preset(preset, width=width)
        self._width_mm: float = spec["width_mm"]
        self._max_height_mm: Optional[float] = spec["max_height_mm"]

        # abc resolution: if user passed None, fall back to preset default.
        self._abc = abc if abc is not None else spec["abc_default"]

        # Initialize label_style from the default + preset's label_size_pt.
        self._label_style: Dict[str, Any] = dict(DEFAULT_LABEL_STYLE)
        self._label_style["size"] = spec["label_size_pt"]

        # PR 5: strict_vectors gates the vector-PDF compositing pipeline's
        # behavior on schematic-load failure.
        self._strict_vectors: bool = bool(strict_vectors)

        # Lazy state — figure is materialized on first finalization trigger.
        self._figure = None  # matplotlib.figure.Figure | None
        self._rows: List[_RowStaging] = []         # staged rows
        # Internal panel storage — populated by _finalize_if_needed.
        # The PUBLIC-ish read accessors (_panels, _panels_ordered) below
        # trigger finalization on access so tests/code that touched the
        # PR 1+2 names continue to work.
        self._panels_dict: Dict[str, Panel] = {}
        self._panels_list: List[Panel] = []
        self._alignments: List = []  # List[_AlignmentRequest]; initialized empty
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # Read-only attributes
    # ------------------------------------------------------------------
    @property
    def width_mm(self) -> float:
        """Canvas width in millimeters."""
        return self._width_mm

    @property
    def figure(self):
        """The underlying matplotlib :class:`Figure`.

        Accessing this property triggers lazy finalization if the
        canvas has at least one staged row but has not yet been
        finalized. Returns ``None`` if no rows have been staged.
        """
        if not self._finalized and self._rows:
            self._finalize_if_needed()
        return self._figure

    @property
    def _panels(self) -> Dict[str, Panel]:
        """Resolved-label-keyed panel dict. Triggers lazy finalization
        if rows are staged but not yet finalized.

        Used internally by ``__getitem__``; PR 1+2 tests also touched
        this attribute directly (via ``canvas._panels``), so the
        accessor preserves that contract.
        """
        if not self._finalized and self._rows:
            self._finalize_if_needed()
        return self._panels_dict

    @property
    def _panels_ordered(self) -> List[Panel]:
        """Insertion-ordered panel list (across all rows). Triggers
        lazy finalization if rows are staged but not yet finalized.
        """
        if not self._finalized and self._rows:
            self._finalize_if_needed()
        return self._panels_list

    @property
    def figure_size_mm(self) -> Optional[Tuple[float, float]]:
        """``(width_mm, height_mm)`` once the figure has been finalized,
        else ``None``.

        Triggers lazy finalization if at least one row has been staged.
        Returns ``None`` only when no rows have been added yet (the
        canvas has not yet committed to any geometry).
        """
        if not self._finalized and self._rows:
            self._finalize_if_needed()
        if self._figure is None:
            return None
        # Convert mpl figure size (inches) back to mm.
        w_in, h_in = self._figure.get_size_inches()
        return (w_in * 25.4, h_in * 25.4)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, key) -> Panel:
        # Empty canvas — preserve KeyError contract from PR 1+2 instead
        # of letting _finalize_if_needed raise RuntimeError.
        if not self._rows and not self._finalized:
            if isinstance(key, int) and not isinstance(key, bool):
                raise KeyError("no panels yet; call add_row() first")
            raise KeyError(
                f"no panel with label {key!r}; "
                f"known labels: []"
            )
        self._finalize_if_needed()
        if isinstance(key, int) and not isinstance(key, bool):
            if not self._panels_list:
                raise KeyError("no panels yet; call add_row() first")
            try:
                return self._panels_list[key]
            except IndexError:
                raise KeyError(
                    f"panel index {key} out of range; "
                    f"only {len(self._panels_list)} panel(s) exist"
                )
        if key not in self._panels_dict:
            raise KeyError(
                f"no panel with label {key!r}; "
                f"known labels: {sorted(self._panels_dict)}"
            )
        return self._panels_dict[key]

    def label_style(self, **kwargs: Any) -> None:
        """Update the canvas-wide label style.

        Accepts any subset of: ``weight``, ``size``, ``family``, ``loc``,
        ``pad_mm``, ``border``, ``bbox``. Missing keys are unchanged.
        ``loc`` must be one of the 8 ultraplot locs (``'ul'``, ``'ur'``,
        ``'ll'``, ``'lr'``, ``'uc'``, ``'lc'``, ``'cl'``, ``'cr'``).
        """
        from publiplots.composer.abc_labels import VALID_LOCS

        if "loc" in kwargs and kwargs["loc"] not in VALID_LOCS:
            raise ValueError(
                f"loc must be one of {sorted(VALID_LOCS)}, got {kwargs['loc']!r}"
            )
        self._label_style.update(kwargs)

    # ------------------------------------------------------------------
    # align — record an explicit alignment override
    # ------------------------------------------------------------------
    def align(
        self,
        panels,
        *,
        edge: str,
        mode: str = "axes",
        anchor=None,
    ) -> None:
        """Record an alignment request to apply at finalize time.

        Parameters
        ----------
        panels : sequence of str
            Panel labels to align. The labels must match panels staged
            via add_row (verbatim str labels; abc-resolved auto-letters
            are NOT yet known at align-time so use canvas[i] indexing
            via the FIRST panel's verbatim label as the anchor if needed).
        edge : str
            One of ``'left'``, ``'right'``, ``'top'``, ``'bottom'``,
            ``'center_x'``, ``'center_y'``, ``'baseline'``.
        mode : str, default ``'axes'``
            ``'axes'`` (axes-bbox edges) or ``'tight'`` (tightbbox edges).
            PR 3 ships axes-bbox primarily; tight uses the same path
            until PR 4.5 lands proper tightbbox measurement.
        anchor : str, optional
            If given, this panel's edge is the reference. Otherwise the
            leftmost / rightmost / topmost / bottommost panel's edge wins
            per edge type.

        Raises
        ------
        RuntimeError
            If the canvas was already finalized.
        ValueError
            If the edge or mode is unknown, or if anchor isn't in panels.
        KeyError
            If any panel label isn't found among staged rows.
        """
        from publiplots.composer.alignment import (
            VALID_EDGES, VALID_MODES, _AlignmentRequest,
        )
        if self._finalized:
            raise RuntimeError(
                "Canvas already finalized; align() must be called before "
                "any figure access (canvas.figure, canvas[...], canvas.savefig)"
            )
        if edge not in VALID_EDGES:
            raise ValueError(
                f"edge must be one of {sorted(VALID_EDGES)}, got {edge!r}"
            )
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
            )
        # Verify the panels are known. Walk staged rows for str labels.
        # (None / False labels aren't addressable by str; users wanting
        # to align them should give them explicit str labels for now.)
        known_labels = set()
        for row in self._rows:
            for p in row.panels:
                if isinstance(p.label, str):
                    known_labels.add(p.label)
        for p in panels:
            if p not in known_labels:
                raise KeyError(
                    f"panel {p!r} not found among staged rows; "
                    f"known str-labeled panels: {sorted(known_labels)}"
                )
        if anchor is not None and anchor not in panels:
            raise ValueError(
                f"anchor must be one of the panels in the request; "
                f"got anchor={anchor!r}, panels={list(panels)}"
            )
        self._alignments.append(_AlignmentRequest(
            panels=tuple(panels),
            edge=edge,
            mode=mode,
            anchor=anchor,
        ))

    # ------------------------------------------------------------------
    # add_row — stages a row; figure creation is deferred to finalize
    # ------------------------------------------------------------------
    def add_row(self, *panels, vpad: float = 4.0) -> None:
        """Add a row of panels to the canvas.

        PR 3 supports calling ``add_row`` multiple times. Rows stack
        top-to-bottom; the first row uses ``vpad=0`` regardless of the
        kwarg; subsequent rows use the user-supplied ``vpad``
        (default 4 mm).

        Lazy finalization: the matplotlib :class:`Figure` is created on
        first access of ``canvas.figure`` / ``canvas[label]`` /
        ``canvas[i]`` / ``canvas.savefig`` (or via an explicit
        :meth:`finalize` call). ``add_row`` itself only stages the row
        record — no matplotlib work happens here.

        Parameters
        ----------
        *panels : PanelAxes (or PanelGrid/PanelText in Tasks 7+)
            One or more panel input records.
        vpad : float, default 4.0
            Vertical pad in mm above this row. Ignored (forced to 0)
            for the FIRST row to avoid extra top padding.

        Raises
        ------
        RuntimeError
            If the canvas was already finalized.
        ValueError, TypeError, ComposerOverflowError
            Same input-validation errors as PR 2's ``add_row``
            (overflow checks defer to finalization).
        """
        if self._finalized:
            raise RuntimeError(
                "Canvas already finalized; add_row() must be called BEFORE "
                "any figure access (canvas.figure, canvas[...], canvas.savefig)"
            )
        self._validate_row_inputs(panels)
        # Eager overflow / flex resolution. We don't STORE the resolved
        # widths here (compute them again in _finalize_if_needed) — this
        # is purely a fail-fast check so users see overflow errors at
        # the offending add_row call, not at finalize time.
        self._check_row_overflow_eager(panels)
        # vpad=0 for the first row — avoids extra top padding.
        effective_vpad = 0.0 if not self._rows else float(vpad)
        self._rows.append(_RowStaging(panels=tuple(panels), vpad_mm=effective_vpad))

    # ------------------------------------------------------------------
    # add_row helpers (private)
    # ------------------------------------------------------------------
    def _validate_row_inputs(self, panels):
        """Zero-panel check, type check, duplicate-label check.

        Walks both already-staged rows and the new ones being added so
        duplicate-label checks span across rows.
        """
        from publiplots.composer.panels import (
            PanelAxes, PanelGrid, PanelImage, PanelText,
        )

        if len(panels) == 0:
            raise ValueError("Canvas.add_row requires at least one panel")
        accepted = (PanelAxes, PanelGrid, PanelText, PanelImage)
        for p in panels:
            if not isinstance(p, accepted):
                raise TypeError(
                    f"add_row panels must be PanelAxes / PanelGrid / PanelText / "
                    f"PanelImage; got {type(p).__name__}"
                )
        # Duplicate label check ACROSS already-staged rows + new ones.
        seen: set = set()
        for row in self._rows:
            for p in row.panels:
                if isinstance(p.label, str):
                    seen.add(p.label)
        for p in panels:
            if isinstance(p.label, str):
                if p.label in seen:
                    raise ValueError(f"duplicate panel label: {p.label!r}")
                seen.add(p.label)

    def _resolve_decorations(self):
        """Read all the rcParams subplots.* values into a dict.

        Returns dict with keys: outer_pad, hpad, title_space, xlabel_space,
        ylabel_space, right. (hpad maps to subplots.wspace.)
        """
        outer_pad = float(resolve_param("subplots.outer_pad", None))
        hpad = float(resolve_param("subplots.wspace", None))  # inter-panel gap
        title_space = float(resolve_param("subplots.title_space", None))
        xlabel_space = float(resolve_param("subplots.xlabel_space", None))
        ylabel_space = float(resolve_param("subplots.ylabel_space", None))
        right = float(resolve_param("subplots.right", None))
        return {
            "outer_pad": outer_pad,
            "hpad": hpad,
            "title_space": title_space,
            "xlabel_space": xlabel_space,
            "ylabel_space": ylabel_space,
            "right": right,
        }

    def _compute_decorations_width(self, decorations, ncols):
        """Compute a row's total decoration width budget.

        ylabel_space and right are PER-COLUMN per the FigureLayout
        contract; outer_pad is once on each side; hpad applies between
        adjacent panels.
        """
        return (
            2 * decorations["outer_pad"]
            + ncols * decorations["ylabel_space"]
            + ncols * decorations["right"]
            + max(ncols - 1, 0) * decorations["hpad"]
        )

    def _check_row_overflow_eager(self, panels):
        """Resolve flex widths + check pinned-row overflow at add_row
        time so users see the error AT the offending call (not deferred
        to finalize).

        Wraps :class:`ValueError` from the resolver into
        :class:`ComposerOverflowError`. Re-raises for finalize to
        recompute (cheaper to recompute than to thread the resolved
        widths through staging state).
        """
        from publiplots.composer._flex import resolve_flex_widths

        decorations = self._resolve_decorations()
        ncols = len(panels)
        raw_widths = tuple(_panel_raw_width(p) for p in panels)
        decorations_width = self._compute_decorations_width(decorations, ncols)
        try:
            col_widths, n_flex = resolve_flex_widths(
                raw_widths,
                canvas_width_mm=self._width_mm,
                decorations_width_mm=decorations_width,
            )
        except ValueError as e:
            pinned_total = sum(float(w) for w in raw_widths if w != "flex")
            requested = pinned_total + decorations_width
            raise ComposerOverflowError(
                str(e),
                requested_mm=requested,
                available_mm=self._width_mm,
            )
        if n_flex == 0:
            panels_width = sum(col_widths)
            requested_width = panels_width + decorations_width
            if requested_width > self._width_mm + 1e-6:
                row_idx = len(self._rows)
                raise ComposerOverflowError(
                    f"row {row_idx} width {requested_width:.2f}mm exceeds "
                    f"canvas width {self._width_mm:.2f}mm; reduce panel "
                    f"widths or use a wider canvas",
                    requested_mm=requested_width,
                    available_mm=self._width_mm,
                )

    # ------------------------------------------------------------------
    # finalize — materialize the matplotlib Figure from staged rows
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Materialize the matplotlib Figure from staged rows.

        Idempotent — a second call is a no-op. Raises ``RuntimeError``
        if no rows have been staged yet.
        """
        self._finalize_if_needed()

    def _finalize_if_needed(self) -> None:
        """Build the matplotlib Figure from the staged rows.

        After this runs, ``self._figure`` is the Figure,
        ``self._panels`` and ``self._panels_ordered`` are populated,
        and ``self._finalized=True``.
        """
        if self._finalized:
            return
        if not self._rows:
            raise RuntimeError(
                "Canvas has no rows yet; call add_row() before finalize()"
            )

        # Per-panel dispatch handles all four panel kinds (PanelAxes,
        # PanelGrid, PanelText, PanelImage) in the loop below.
        from publiplots.composer.panels import (
            PanelAxes, PanelGrid, PanelImage, PanelText,
        )

        # Resolve decorations once (shared across rows).
        decorations = self._resolve_decorations()

        # Per-row: resolve flex, check overflow, compute row_height.
        from publiplots.composer._flex import resolve_flex_widths
        rows_for_geometry = []
        for r_idx, row in enumerate(self._rows):
            ncols = len(row.panels)
            raw_widths = tuple(_panel_raw_width(p) for p in row.panels)
            row_height = max(_panel_raw_height(p) for p in row.panels)
            decorations_width = self._compute_decorations_width(
                decorations, ncols
            )
            try:
                col_widths, n_flex = resolve_flex_widths(
                    raw_widths,
                    canvas_width_mm=self._width_mm,
                    decorations_width_mm=decorations_width,
                )
            except ValueError as e:
                pinned_total = sum(float(w) for w in raw_widths if w != "flex")
                requested = pinned_total + decorations_width
                raise ComposerOverflowError(
                    str(e),
                    requested_mm=requested,
                    available_mm=self._width_mm,
                )
            if n_flex == 0:
                panels_width = sum(col_widths)
                requested_width = panels_width + decorations_width
                if requested_width > self._width_mm + 1e-6:
                    raise ComposerOverflowError(
                        f"row {r_idx} width {requested_width:.2f}mm exceeds "
                        f"canvas width {self._width_mm:.2f}mm; reduce panel "
                        f"widths or use a wider canvas",
                        requested_mm=requested_width,
                        available_mm=self._width_mm,
                    )
            rows_for_geometry.append({
                "panel_widths_mm": col_widths,
                "row_height_mm": row_height,
                "vpad_mm": row.vpad_mm,
            })

        # Compute multi-row geometry.
        from publiplots.composer._layout import compute_canvas_geometry
        geometry = compute_canvas_geometry(
            rows=rows_for_geometry,
            canvas_width_mm=self._width_mm,
            outer_pad=decorations["outer_pad"],
            ylabel_space=decorations["ylabel_space"],
            right=decorations["right"],
            wspace=decorations["hpad"],
            title_space=decorations["title_space"],
            xlabel_space=decorations["xlabel_space"],
        )

        # Apply alignment requests (if any). Each shifts panel rects
        # WITHIN their original slot bounds; raises ComposerAlignmentError
        # if a shift would exit the slot.
        if self._alignments:
            from publiplots.composer.alignment import apply_alignments
            from publiplots.composer._layout import CanvasGeometry

            # Build dict[label, rect_mm] for label-keyed lookup, plus
            # the inviolate slot dict. Both start identical (the slot
            # IS the panel's natural rect from geometry).
            panel_rects_mm = {}
            for r_idx, (row, rects_mm) in enumerate(
                zip(self._rows, geometry.row_axes_rects_mm)
            ):
                for c_idx, (panel_input, rect_mm) in enumerate(
                    zip(row.panels, rects_mm)
                ):
                    if isinstance(panel_input.label, str):
                        panel_rects_mm[panel_input.label] = rect_mm
            slot_rects_mm = dict(panel_rects_mm)

            updated_rects = apply_alignments(
                requests=self._alignments,
                panel_rects_mm=panel_rects_mm,
                slot_rects_mm=slot_rects_mm,
            )

            # Re-thread updated rects back into geometry.row_axes_rects_mm
            # by label match.
            new_row_rects = []
            for r_idx, (row, rects_mm) in enumerate(
                zip(self._rows, geometry.row_axes_rects_mm)
            ):
                row_new = []
                for c_idx, (panel_input, rect_mm) in enumerate(
                    zip(row.panels, rects_mm)
                ):
                    if (
                        isinstance(panel_input.label, str)
                        and panel_input.label in updated_rects
                    ):
                        row_new.append(updated_rects[panel_input.label])
                    else:
                        row_new.append(rect_mm)
                new_row_rects.append(row_new)
            # CanvasGeometry is frozen; rebuild it with the updated rects.
            geometry = CanvasGeometry(
                canvas_width_mm=geometry.canvas_width_mm,
                canvas_height_mm=geometry.canvas_height_mm,
                row_axes_rects_mm=new_row_rects,
            )

        # Create the matplotlib figure at the computed mm size.
        import matplotlib.pyplot as plt
        _MM2INCH = 1.0 / 25.4
        fig = plt.figure(
            figsize=(
                geometry.canvas_width_mm * _MM2INCH,
                geometry.canvas_height_mm * _MM2INCH,
            ),
            layout=None,
        )
        self._figure = fig

        # Resolve labels FLAT across all rows (abc spans rows).
        from publiplots.composer.abc_labels import (
            resolve_labels, merge_label_style, render_label,
        )
        flat_panels = []
        for row in self._rows:
            for p in row.panels:
                flat_panels.append(p)
        raw_labels = [p.label for p in flat_panels]
        resolved_labels = resolve_labels(panel_labels=raw_labels, abc=self._abc)

        # Per-panel: dispatch by panel input type.
        flat_idx = 0
        for r_idx, (row, rects_mm) in enumerate(
            zip(self._rows, geometry.row_axes_rects_mm)
        ):
            for c_idx, (panel_input, rect_mm) in enumerate(
                zip(row.panels, rects_mm)
            ):
                x_mm, y_mm, w_mm, h_mm = rect_mm
                resolved_label = resolved_labels[flat_idx]
                resolved_style = merge_label_style(
                    self._label_style, panel_input.label_style
                )

                if isinstance(panel_input, PanelGrid):
                    # Build the inner sub-grid of axes inside the panel's
                    # outer mm rect.
                    inner_axes = self._build_panel_grid_axes(
                        panel_input, fig, rect_mm,
                        geometry.canvas_width_mm, geometry.canvas_height_mm,
                    )
                    panel = Panel(
                        label=resolved_label,
                        kind="axesgrid",
                        ax=None,
                        size_mm=(w_mm, h_mm),
                        bbox_mm=(x_mm, y_mm, w_mm, h_mm),
                        resolved_label_style=resolved_style,
                        axes=inner_axes,
                    )
                    # Render the abc label on the TOP-LEFT inner axes
                    # (the panel's "first" cell visually).
                    label_target_ax = inner_axes[0, 0]
                elif isinstance(panel_input, PanelText):
                    rect_frac = (
                        x_mm / geometry.canvas_width_mm,
                        y_mm / geometry.canvas_height_mm,
                        w_mm / geometry.canvas_width_mm,
                        h_mm / geometry.canvas_height_mm,
                    )
                    ax = self._build_panel_text_axes(
                        panel_input, fig, rect_frac,
                    )
                    panel = Panel(
                        label=resolved_label,
                        kind="text",
                        ax=ax,
                        size_mm=(w_mm, h_mm),
                        bbox_mm=(x_mm, y_mm, w_mm, h_mm),
                        resolved_label_style=resolved_style,
                        axes=None,
                    )
                    label_target_ax = ax
                elif isinstance(panel_input, PanelImage):
                    # PanelImage: hidden axes reserves the slot during
                    # matplotlib's render. The actual schematic gets
                    # stamped POST-savefig by the PDF compositing pipeline
                    # (see compositing/pdf.py).
                    rect_frac = (
                        x_mm / geometry.canvas_width_mm,
                        y_mm / geometry.canvas_height_mm,
                        w_mm / geometry.canvas_width_mm,
                        h_mm / geometry.canvas_height_mm,
                    )
                    ax = self._build_panel_image_axes(fig, rect_frac)
                    panel = Panel(
                        label=resolved_label,
                        kind="image",
                        ax=ax,
                        size_mm=(w_mm, h_mm),
                        bbox_mm=(x_mm, y_mm, w_mm, h_mm),
                        resolved_label_style=resolved_style,
                        axes=None,
                        image_path=panel_input.path,
                        image_align=panel_input.align,
                        image_clip=panel_input.clip,
                    )
                    label_target_ax = ax
                else:
                    # PanelAxes.
                    rect_frac = (
                        x_mm / geometry.canvas_width_mm,
                        y_mm / geometry.canvas_height_mm,
                        w_mm / geometry.canvas_width_mm,
                        h_mm / geometry.canvas_height_mm,
                    )
                    ax = fig.add_axes(rect_frac)
                    panel = Panel(
                        label=resolved_label,
                        kind="axes",
                        ax=ax,
                        size_mm=(w_mm, _panel_raw_height(panel_input)),
                        bbox_mm=(x_mm, y_mm, w_mm, h_mm),
                        resolved_label_style=resolved_style,
                        axes=None,
                    )
                    label_target_ax = ax

                if isinstance(resolved_label, str) and resolved_label:
                    render_label(label_target_ax, resolved_label, style=resolved_style)
                    self._panels_dict[resolved_label] = panel
                self._panels_list.append(panel)
                flat_idx += 1

        # Reactor: ONLY for single-row canvases. Multi-row canvases
        # skip the reactor (use static rcParams reservations) — refined
        # in PR 4.5 if user feedback demands per-row auto-grow.
        if len(self._rows) == 1:
            from publiplots.layout.figure_layout import FigureLayout
            from publiplots.layout.auto_layout import SubplotsAutoLayout
            ncols = len(self._rows[0].panels)
            col_widths = tuple(rect[2] for rect in geometry.row_axes_rects_mm[0])
            row_height = max(_panel_raw_height(p) for p in self._rows[0].panels)
            layout = FigureLayout(
                nrows=1,
                ncols=ncols,
                axes_size=(col_widths[0], row_height),
                col_widths=col_widths,
                row_heights=(row_height,),
                title_space=(decorations["title_space"],),
                xlabel_space=(decorations["xlabel_space"],),
                ylabel_space=(decorations["ylabel_space"],) * ncols,
                right=(decorations["right"],) * ncols,
                hspace=0.0,
                wspace=decorations["hpad"],
                outer_pad=decorations["outer_pad"],
                legend_column=0.0,
                suptitle_space=0.0,
            )
            fig._publiplots_auto_layout = SubplotsAutoLayout(
                fig, layout, locked=set(), locked_positions={}
            )

        self._finalized = True

    # ------------------------------------------------------------------
    # PanelGrid construction helpers (PR 3 Task 7)
    # ------------------------------------------------------------------
    def _build_panel_grid_axes(
        self,
        panel_grid,
        fig,
        outer_rect_mm,
        canvas_w_mm,
        canvas_h_mm,
    ):
        """Lay out the inner axes grid inside the panel's mm rect.

        Row 0 sits at the TOP of the panel; the topmost row therefore has
        the LARGEST y in matplotlib's bottom-left-origin coordinates.
        ``axes_size`` is the per-cell mm size; ``hspace``/``wspace``
        separate cells.

        Returns
        -------
        axes : numpy.ndarray of shape (nrows, ncols), dtype=object
            Array of matplotlib Axes objects.
        """
        import numpy as np

        nr, nc = panel_grid.shape
        cell_w, cell_h = panel_grid.axes_size
        hspace, wspace = panel_grid.hspace, panel_grid.wspace

        x0_mm, y0_mm, _w_mm, _h_mm = outer_rect_mm

        axes = np.empty((nr, nc), dtype=object)
        for r in range(nr):
            for c in range(nc):
                cell_x_mm = x0_mm + c * (cell_w + wspace)
                # Row 0 sits at the TOP of the panel; in bottom-left
                # origin coords the topmost row has the LARGEST y.
                cell_y_mm = y0_mm + (nr - 1 - r) * (cell_h + hspace)
                rect_frac = (
                    cell_x_mm / canvas_w_mm,
                    cell_y_mm / canvas_h_mm,
                    cell_w / canvas_w_mm,
                    cell_h / canvas_h_mm,
                )
                share_x = self._resolve_panel_grid_share(
                    panel_grid.sharex, axes, r, c, axis="x"
                )
                share_y = self._resolve_panel_grid_share(
                    panel_grid.sharey, axes, r, c, axis="y"
                )
                kwargs = {}
                if share_x is not None:
                    kwargs["sharex"] = share_x
                if share_y is not None:
                    kwargs["sharey"] = share_y
                axes[r, c] = fig.add_axes(rect_frac, **kwargs)
        return axes

    @staticmethod
    def _resolve_panel_grid_share(share, axes, r, c, *, axis):
        """Return the axes to share with, or None.

        Mirrors :func:`publiplots.layout.subplots._resolve_shared`.
        ``share`` has already been validated by :class:`PanelGrid`.
        """
        if share in (False, "none"):
            return None
        if r == 0 and c == 0:
            return None
        if share in (True, "all"):
            return axes[0, 0]
        if share == "row":
            return axes[r, 0] if c > 0 else None
        if share == "col":
            return axes[0, c] if r > 0 else None
        # Unreachable — PanelGrid validates share values up-front.
        return None

    # ------------------------------------------------------------------
    # PanelText construction helper (PR 3 Task 8)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_panel_text_axes(panel_text, fig, rect_frac):
        """Create a hidden axes for a text panel and place the text artist.

        The text is centered at (0.5, 0.5) in axes-fraction with
        ha='center', va='center' by default; user-supplied ``text_kw``
        can override these.
        """
        ax = fig.add_axes(rect_frac)
        ax.set_axis_off()
        ax.patch.set_visible(False)
        # set_axis_off() sets axison=False (skips drawing) but does NOT
        # flip per-spine visibility flags; explicitly hide spines so
        # introspection (and any save path that bypasses axison) sees a
        # clean text-only axes.
        for spine in ax.spines.values():
            spine.set_visible(False)

        text_kw = {"ha": "center", "va": "center"}
        text_kw.update(panel_text.text_kw or {})
        ax.text(0.5, 0.5, panel_text.text, transform=ax.transAxes, **text_kw)
        return ax

    # ------------------------------------------------------------------
    # PanelImage construction helper (PR 5)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_panel_image_axes(fig, rect_frac):
        """Create a hidden axes that reserves a slot for a PanelImage.

        The schematic itself is stamped post-savefig by the PDF
        compositing pipeline (see :mod:`compositing.pdf`). This axes
        only reserves the mm rect during matplotlib's render to PDF
        buffer.
        """
        ax = fig.add_axes(rect_frac)
        ax.set_axis_off()
        ax.patch.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    # ------------------------------------------------------------------
    # save_multiple — write to many formats from one canvas
    # ------------------------------------------------------------------
    def save_multiple(
        self,
        stem,
        formats: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List:
        """Save the canvas to multiple formats from a shared stem.

        Sugar over a Python loop ``for ext in formats: self.savefig(...)``
        that pre-validates the ext list to avoid partial writes on
        invalid input. The default ``formats=['png', 'pdf']`` mirrors
        the free function :func:`publiplots.save_multiple`.

        Parameters
        ----------
        stem : str or Path
            Output stem. The format extension is appended via
            ``Path(stem).with_suffix(f".{ext}")``. If ``stem`` already
            has a suffix it is REPLACED (matching ``Path.with_suffix``
            semantics) — so ``save_multiple('figure.draft', ['pdf'])``
            writes ``figure.pdf``, not ``figure.draft.pdf``.
        formats : sequence of str, optional
            Format names. Defaults to ``['png', 'pdf']`` (parity with
            ``pp.save_multiple``). Each entry is a known raster ext
            (``png``/``jpg``/``jpeg``/``tif``/``tiff``) or vector ext
            (``pdf``/``svg``); leading dots are accepted (``.pdf`` is
            equivalent to ``pdf``).
        **kwargs
            Forwarded to :meth:`savefig` for every format. ``cmyk=True``
            is pre-validated against the format list — passing
            ``cmyk=True + formats=['tif', 'pdf']`` raises ``ValueError``
            BEFORE writing the ``.tif`` so there's no partial state from
            cmyk-misuse. Other failure modes (invalid kwarg, disk full,
            permission denied) cannot be pre-validated and may leave
            earlier-iteration files written when a later iteration raises.

        Returns
        -------
        list of Path
            The paths actually written, in the same order as ``formats``.

        Raises
        ------
        ValueError
            ``formats`` is empty, contains duplicates, contains a
            non-string entry, contains an unknown ext, or ``cmyk=True``
            is paired with a vector ext.

        Notes
        -----
        Divergence from the free function ``pp.save_multiple``: the
        free function is permissive (forwards bad ext through to
        matplotlib's writers, which raise late). This method is strict
        because it knows the exact set of supported exts and can give
        users a better error message up-front.
        """
        from publiplots.composer._save import (
            _RASTER_EXTS, _VECTOR_PDF_EXTS, _VECTOR_SVG_EXTS,
        )
        from pathlib import Path as _Path

        if formats is None:
            formats = ["png", "pdf"]
        if not formats:
            raise ValueError(
                "save_multiple: formats must be a non-empty sequence; "
                f"got {formats!r}"
            )
        # Validate every entry is a string.
        for f in formats:
            if not isinstance(f, str):
                raise ValueError(
                    f"save_multiple: formats entries must be str; "
                    f"got {f!r} of type {type(f).__name__}"
                )
        # Normalize: strip a leading dot, lowercase.
        normalized = [f.lstrip(".").lower() for f in formats]
        # Reject duplicates AFTER normalization so '.pdf' + 'pdf' is one.
        seen: set = set()
        for n in normalized:
            if n in seen:
                raise ValueError(
                    f"save_multiple: duplicate format {n!r} in {list(formats)!r}"
                )
            seen.add(n)
        # Validate each ext is known.
        known = (_RASTER_EXTS | _VECTOR_PDF_EXTS | _VECTOR_SVG_EXTS)
        for n in normalized:
            ext = f".{n}"
            if ext not in known:
                raise ValueError(
                    f"save_multiple: unsupported format {n!r}; "
                    f"supported: {sorted(e.lstrip('.') for e in known)}"
                )
        # Pre-validate cmyk vs vector exts. We MUST catch this before
        # writing the first file so a multi-format save doesn't leave
        # partial state.
        if kwargs.get("cmyk", False):
            non_raster = [
                n for n in normalized
                if f".{n}" not in _RASTER_EXTS
            ]
            if non_raster:
                raise ValueError(
                    f"save_multiple: cmyk=True is only valid for raster "
                    f"outputs (.tif/.tiff/.jpg/.jpeg); got "
                    f"non-raster formats {non_raster!r} in formats list. "
                    f"Drop the vector formats or split into two calls."
                )

        # Build paths, then iterate self.savefig.
        out: List[_Path] = []
        for n in normalized:
            p = _Path(stem).with_suffix(f".{n}")
            self.savefig(p, **kwargs)
            out.append(p)
        return out

    # ------------------------------------------------------------------
    # embed_figure — post-staging: attach a Figure into a PanelImage slot
    # ------------------------------------------------------------------
    def embed_figure(self, label, figure) -> None:
        """Attach a matplotlib :class:`~matplotlib.figure.Figure` into a
        previously-staged PanelImage slot.

        The Figure is rendered to a deterministic PDF/SVG byte buffer at
        compose time; the existing ``compositing.pdf`` / ``compositing.svg``
        orchestrators treat it like a vector schematic source. The
        target panel MUST be ``kind='image'`` (a :class:`PanelImage`)
        AND not yet have an embedded figure (``embed_figure`` is one-shot
        per panel — silent overwrites are rejected).

        Parameters
        ----------
        label : str or int
            Panel resolver. Accepts the same values as
            :meth:`Canvas.__getitem__`: a ``str`` resolved label
            (post abc-resolution) or an ``int`` insertion index.
        figure : matplotlib.figure.Figure
            The figure to embed. Duck-typed: any object that responds to
            ``figure.savefig(buf, format='pdf'|'svg', ...)`` is accepted.
            ``embed_figure`` does NOT validate the type; the compositing
            pipeline calls ``figure.savefig`` and surfaces TypeErrors
            from non-Figures naturally.

        Raises
        ------
        RuntimeError
            If the canvas has no rows yet (call ``add_row`` first); OR
            if the resolved panel already has an embedded figure
            (``embed_figure`` is one-shot per panel).
        KeyError
            If ``label`` is not found among staged panels.
        TypeError
            If the resolved panel is not ``kind='image'``
            (``embed_figure`` only attaches to PanelImage slots).

        Notes
        -----
        - ``embed_figure`` stores a *reference* to the Figure (not a
          snapshot). Mutating the Figure between ``embed_figure`` and
          ``savefig`` will be reflected in the rendered output. Don't
          call ``fig.tight_layout()`` (per project rule) and don't
          ``fig.clear()`` it.
        - The pairing constraint with PanelImage is documented on
          :class:`PanelImage` itself: a no-path ``PanelImage`` MUST be
          paired with ``embed_figure`` before ``savefig``, else the
          composer raises :class:`ComposerVectorError`. ``embed_figure``
          on a path-bearing PanelImage is also legal — the embedded
          figure wins over the path.
        """
        # Empty canvas → preserve the same RuntimeError semantics that
        # _finalize_if_needed raises on an empty canvas. This is the
        # documented "embed_figure before any add_row" branch.
        if not self._rows and not self._finalized:
            raise RuntimeError(
                "Canvas has no rows yet; call add_row() before embed_figure()"
            )
        # Trigger lazy finalization so the Panel records exist.
        self._finalize_if_needed()
        # Resolve via the existing __getitem__ contract — same KeyError
        # contract for str / int / out-of-range.
        panel = self[label]
        # Validate kind.
        if panel.kind != "image":
            raise TypeError(
                f"embed_figure: target panel {label!r} is "
                f"kind={panel.kind!r}, not 'image'; embed_figure only "
                f"attaches to PanelImage slots."
            )
        # One-shot per panel — silent overwrites would mask bugs.
        if panel.embedded_figure is not None:
            raise RuntimeError(
                f"embed_figure: panel {label!r} already has an embedded "
                f"figure; embed_figure is one-shot per panel. Stage a "
                f"fresh Canvas + PanelImage if you need to retarget."
            )
        # Mutate via object.__setattr__ — Panel is a frozen dataclass.
        object.__setattr__(panel, "embedded_figure", figure)

    # ------------------------------------------------------------------
    # savefig — raster only (vector lands in PR 5/PR 6)
    # ------------------------------------------------------------------
    def savefig(
        self,
        path,
        *,
        cmyk: bool = False,
        tiff_compression: str = "tiff_lzw",
        external_raster: bool = False,
        **kwargs,
    ) -> None:
        """Save the canvas to a file.

        Triggers lazy finalization if the canvas has at least one
        staged row but has not yet been finalized.

        Supports raster formats (PNG, JPG, TIFF) plus vector PDF (PR 5)
        and vector SVG (PR 6a). For vector outputs, any PanelImage
        slots are composited via the appropriate pipeline (pypdf for
        PDF, lxml for SVG).

        Parameters
        ----------
        path : str or Path
            Output file path. Extension determines the format.
        cmyk : bool, default False
            Convert RGB→CMYK on raster output. Valid only for
            ``.tif``/``.tiff``/``.jpg``/``.jpeg``; pairing with PDF /
            SVG / PNG raises ``ValueError``. Uses Pillow's basic
            sRGB→CMYK conversion (no ICC profile in PR 6b — emits a
            ``UserWarning``).
        tiff_compression : str, default ``'tiff_lzw'``
            TIFF compression knob. Other values: ``'tiff_deflate'``,
            ``'raw'``, ``'tiff_jpeg'``, etc (Pillow's compression
            vocab). Ignored for non-TIFF exts.
        external_raster : bool, default False
            For SVG outputs: when ``True``, write raster sources to
            sidecar PNG files rather than inline base64 data-URIs.
            Sidecar files are named ``{stem}-{idx}-{label}.png`` next to
            the SVG; repeated saves with the same stem overwrite the
            sidecars (idempotent re-render). Silent no-op for non-SVG
            outputs.
        **kwargs
            Forwarded to :func:`publiplots.savefig` (raster) or to the
            vector compositor (PDF/SVG).

        Raises
        ------
        RuntimeError
            If :meth:`add_row` has not been called yet.
        ValueError
            If the path's extension is not a known raster or vector type;
            or ``cmyk=True`` is paired with a non-CMYK-capable format.
        ComposerVectorError
            If a vector schematic fails to load and ``strict_vectors=True``.
        """
        if not self._rows and not self._finalized:
            raise RuntimeError(
                "Canvas has no figure yet; call add_row() before savefig()"
            )
        # PR 6b: pre-validate cmyk vs ext at the canvas layer (the dispatch
        # layer also validates defensively, but a clear error message
        # naming the ext is friendlier here). We do this BEFORE
        # _finalize_if_needed so a bad cmyk argument doesn't materialize
        # the figure unnecessarily.
        from pathlib import Path as _Path
        ext = _Path(path).suffix.lower()
        if cmyk:
            if ext == ".png":
                raise ValueError(
                    "PNG does not support CMYK; use .tif/.tiff/.jpg/.jpeg "
                    "instead."
                )
            if ext in {".pdf", ".svg"}:
                raise ValueError(
                    f"cmyk=True is only valid for raster outputs "
                    f"(.tif/.tiff/.jpg/.jpeg); got {ext!r}. "
                    f"matplotlib's PDF/SVG backends emit RGB; convert "
                    f"the matching raster output instead."
                )
        self._finalize_if_needed()
        from publiplots.composer._save import dispatch_savefig
        dispatch_savefig(
            self._figure, path,
            panels=self._panels_list,
            strict_vectors=self._strict_vectors,
            cmyk=cmyk,
            tiff_compression=tiff_compression,
            external_raster=external_raster,
            **kwargs,
        )
