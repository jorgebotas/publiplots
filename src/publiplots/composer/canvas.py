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
from publiplots.composer.panels import Panel, PanelAxes
from publiplots.composer.presets import resolve_preset
from publiplots.themes.rcparams import resolve_param


def _panel_raw_width(panel) -> Any:
    """Return the panel's raw width input for flex resolution.

    - PanelAxes / PanelText: ``panel.size[0]`` (float or ``'flex'`` sentinel)
    - PanelGrid: ``panel.size_mm[0]`` (computed grid outer width; never flex)

    PanelGrid has no ``size`` attribute — its outer width is computed
    from ``shape`` + ``axes_size`` + spacing via the ``size_mm`` property.
    Reading ``panel.size[0]`` unconditionally would AttributeError on
    PanelGrid inputs to :meth:`Canvas.add_row`.
    """
    from publiplots.composer.panels import PanelGrid
    if isinstance(panel, PanelGrid):
        return panel.size_mm[0]
    # PanelAxes and PanelText both expose size: (w_or_flex, h)
    return panel.size[0]


def _panel_raw_height(panel) -> float:
    """Return the panel's height in mm.

    - PanelAxes / PanelText: ``panel.size[1]``
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
    Vector PDF/SVG save dispatches land in PR 5/PR 6.
    ``canvas.savefig('fig.pdf')`` raises :class:`NotImplementedError`
    in the current release (until PR 5/PR 6 land).
    """

    def __init__(
        self,
        preset: str,
        *,
        width: Optional[float] = None,
        abc: Union[str, bool, Sequence[str], None] = None,
    ) -> None:
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

        # Lazy state — figure is materialized on first finalization trigger.
        self._figure = None  # matplotlib.figure.Figure | None
        self._rows: List[_RowStaging] = []         # staged rows
        # Internal panel storage — populated by _finalize_if_needed.
        # The PUBLIC-ish read accessors (_panels, _panels_ordered) below
        # trigger finalization on access so tests/code that touched the
        # PR 1+2 names continue to work.
        self._panels_dict: Dict[str, Panel] = {}
        self._panels_list: List[Panel] = []
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
        from publiplots.composer.panels import PanelAxes, PanelGrid, PanelText

        if len(panels) == 0:
            raise ValueError("Canvas.add_row requires at least one panel")
        accepted = (PanelAxes, PanelGrid, PanelText)
        for p in panels:
            if not isinstance(p, accepted):
                raise TypeError(
                    f"Canvas.add_row accepts PanelAxes, PanelGrid, or PanelText "
                    f"in PR 3, got {type(p).__name__} (PanelImage lands in PR 5)"
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

        # Reject PanelGrid/PanelText for now (Tasks 7+8 will add them).
        from publiplots.composer.panels import PanelAxes, PanelGrid, PanelText
        for r_idx, row in enumerate(self._rows):
            for p in row.panels:
                if isinstance(p, PanelGrid):
                    raise NotImplementedError(
                        f"PanelGrid construction lands in PR3 Task 7 "
                        f"(found PanelGrid in row {r_idx})"
                    )
                if isinstance(p, PanelText):
                    raise NotImplementedError(
                        f"PanelText construction lands in PR3 Task 8 "
                        f"(found PanelText in row {r_idx})"
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

        # Per-panel: create axes, register Panel.
        flat_idx = 0
        for r_idx, (row, rects_mm) in enumerate(
            zip(self._rows, geometry.row_axes_rects_mm)
        ):
            for c_idx, (panel_input, rect_mm) in enumerate(
                zip(row.panels, rects_mm)
            ):
                x_mm, y_mm, w_mm, h_mm = rect_mm
                rect_frac = (
                    x_mm / geometry.canvas_width_mm,
                    y_mm / geometry.canvas_height_mm,
                    w_mm / geometry.canvas_width_mm,
                    h_mm / geometry.canvas_height_mm,
                )
                ax = fig.add_axes(rect_frac)

                resolved_label = resolved_labels[flat_idx]
                resolved_style = merge_label_style(
                    self._label_style, panel_input.label_style
                )
                # The flex-resolved width comes from the geometry rect.
                col_width_resolved = w_mm
                panel = Panel(
                    label=resolved_label,
                    kind="axes",
                    ax=ax,
                    size_mm=(col_width_resolved, _panel_raw_height(panel_input)),
                    bbox_mm=(x_mm, y_mm, w_mm, h_mm),
                    resolved_label_style=resolved_style,
                )

                if isinstance(resolved_label, str) and resolved_label:
                    render_label(ax, resolved_label, style=resolved_style)
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
    # savefig — raster only (vector lands in PR 5/PR 6)
    # ------------------------------------------------------------------
    def savefig(self, path, **kwargs) -> None:
        """Save the canvas to a file.

        Triggers lazy finalization if the canvas has at least one
        staged row but has not yet been finalized.

        Currently supports raster formats (PNG, JPG, TIFF). PDF and SVG
        raise :class:`NotImplementedError` until PR 5 / PR 6 land the
        vector compositing pipelines.

        Parameters
        ----------
        path : str or Path
            Output file path. Extension determines the format.
        **kwargs
            Forwarded to :func:`publiplots.savefig`.

        Raises
        ------
        RuntimeError
            If :meth:`add_row` has not been called yet.
        NotImplementedError
            If ``path`` ends in ``.pdf`` (PR 5) or ``.svg`` (PR 6).
        ValueError
            If the path's extension is not a known raster or vector type.
        """
        if not self._rows and not self._finalized:
            raise RuntimeError(
                "Canvas has no figure yet; call add_row() before savefig()"
            )
        self._finalize_if_needed()
        from publiplots.composer._save import dispatch_savefig
        dispatch_savefig(self._figure, path, **kwargs)
